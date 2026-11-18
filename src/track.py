import os
import sys
import tempfile
import cv2
import numpy as np
import torch

class BaseTracker:
    def initialize(self, first_frame, first_bbox):
        """
        Initialize the tracker.
        first_frame: numpy array (H, W, 3) in RGB format.
        first_bbox: tuple/list (x, y, width, height) in pixels.
        """
        raise NotImplementedError

    def track(self, next_frame):
        """
        Track the target in the next frame.
        next_frame: numpy array (H, W, 3) in RGB format.
        Returns:
            mask: boolean mask (H, W) or None.
            bbox: tuple (x, y, width, height) of predicted bounding box.
            confidence: float tracking confidence score.
        """
        raise NotImplementedError

class SamuraiTracker(BaseTracker):
    def __init__(self, model_size="large", device="cuda"):
        """
        model_size: "base" (base_plus) or "large"
        device: PyTorch device
        """
        self.device = torch.device(device)
        self.model_size = model_size
        self.use_cuda = self.device.type == "cuda"

        # Prefer SAMURAI_REPO_DIR; fall back to the old Azure path for compatibility
        self.repo_dir = os.environ.get(
            "SAMURAI_REPO_DIR",
            "/home/azureuser/Nikhil/samurai_project/hotc2026_run/samurai/sam2",
        )
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)

        from sam2.build_sam import build_sam2_video_predictor

        import sam2.sam2_video_predictor
        import sam2.utils.misc
        sam2.sam2_video_predictor.tqdm = lambda x, *args, **kwargs: x
        sam2.utils.misc.tqdm = lambda x, *args, **kwargs: x

        if model_size == "large":
            cfg_path = "configs/samurai/sam2.1_hiera_l.yaml"
            ckpt_path = os.path.join(self.repo_dir, "checkpoints/sam2.1_hiera_large.pt")
        elif model_size == "base":
            cfg_path = "configs/samurai/sam2.1_hiera_b+.yaml"
            ckpt_path = os.path.join(self.repo_dir, "checkpoints/sam2.1_hiera_base_plus.pt")
        else:
            raise ValueError(f"Unsupported model size: {model_size}")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        self.predictor = build_sam2_video_predictor(cfg_path, ckpt_path, device=self.device)

        self.state = None
        self.current_frame_idx = 0
        self.last_bbox = None
        self.tmp_dir_holder = None
        self.orig_h = None
        self.orig_w = None

    def _autocast(self):
        if self.use_cuda:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return torch.autocast("cpu", dtype=torch.float32)

    def initialize(self, first_frame, first_bbox):
        if self.tmp_dir_holder is not None:
            try:
                self.tmp_dir_holder.cleanup()
            except Exception:
                pass

        self.orig_h, self.orig_w = first_frame.shape[:2]
        self.last_bbox = tuple(first_bbox)

        self.tmp_dir_holder = tempfile.TemporaryDirectory(prefix="samurai_init_")
        tmp_path = self.tmp_dir_holder.name

        bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(tmp_path, "00000.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        with self._autocast():
            self.state = self.predictor.init_state(
                video_path=tmp_path,
                offload_video_to_cpu=True,
                offload_state_to_cpu=False,
            )

            x, y, w, h = first_bbox
            box_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
            _, obj_ids, mask_logits = self.predictor.add_new_points_or_box(
                self.state, box=box_xyxy, frame_idx=0, obj_id=0
            )

        self.current_frame_idx = 0

    def track(self, next_frame):
        self.current_frame_idx += 1

        img_sz = self.predictor.image_size
        resized = cv2.resize(next_frame, (img_sz, img_sz), interpolation=cv2.INTER_LINEAR)

        resized_float = resized.astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(resized_float).permute(2, 0, 1)

        img_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame_tensor = (frame_tensor - img_mean) / img_std

        target_device = self.state["images"].device
        frame_tensor = frame_tensor.to(target_device)

        self.state["images"] = torch.cat([self.state["images"], frame_tensor.unsqueeze(0)], dim=0)
        self.state["num_frames"] += 1

        mask_logits_out = None

        with self._autocast():
            for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(
                self.state,
                start_frame_idx=self.current_frame_idx,
                max_frame_num_to_track=1,
            ):
                if frame_idx == self.current_frame_idx:
                    mask_logits_out = mask_logits

        if mask_logits_out is not None:
            mask = (mask_logits_out[0] > 0.0).squeeze().cpu().numpy()

            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                all_pts = np.concatenate(contours)
                rx, ry, rw, rh = cv2.boundingRect(all_pts)

                rx = min(max(0, rx), self.orig_w - 2)
                ry = min(max(0, ry), self.orig_h - 2)
                rw = min(max(2, rw), self.orig_w - rx)
                rh = min(max(2, rh), self.orig_h - ry)

                bbox = (float(rx), float(ry), float(rw), float(rh))
                self.last_bbox = bbox
            else:
                bbox = self.last_bbox
        else:
            mask = np.zeros((self.orig_h, self.orig_w), dtype=bool)
            bbox = self.last_bbox

        return mask, bbox, 1.0

def get_tracker(tracker_name, device="cuda"):
    if tracker_name == "samurai_base":
        return SamuraiTracker(model_size="base", device=device)
    elif tracker_name == "samurai_large":
        return SamuraiTracker(model_size="large", device=device)
    else:
        raise ValueError(f"Unknown tracker: {tracker_name}")
