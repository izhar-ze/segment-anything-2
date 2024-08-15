import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import torch
from typing import List, Tuple, Optional, Union, Dict, Any, Iterator

from sam2 import build_sam, sam2_image_predictor
from sam2.build_sam import build_sam2_video_predictor


class SAM2Error(Exception):
    """Custom exception class for SAM2-related errors."""
    pass


class ImagePredictor:
    def __init__(self, sam_model, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0):
        self.predictor = sam2_image_predictor.SAM2ImagePredictor(
            sam_model, mask_threshold, max_hole_area, max_sprinkle_area
        )

    def set_image(self, image: Union[np.ndarray, Image.Image]) -> None:
        self.predictor.set_image(image)

    def predict(self, *args, **kwargs):
        return self.predictor.predict(*args, **kwargs)


class SAM2Model:
    def __init__(self, video_dir, model_cfg, sam2_checkpoint, device="cuda", mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0):
        self.video_dir = video_dir
        self.model_cfg = model_cfg
        self.sam2_checkpoint = sam2_checkpoint
        self.device = device
        self._setup_device()
        self.sam2 = build_sam.build_sam2(model_cfg, sam2_checkpoint)
        self.image_predictor = ImagePredictor(self.sam2, mask_threshold, max_hole_area, max_sprinkle_area)
        self.video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        self.frame_names = self._get_frame_names()
        self.video_segments = {}
        self.video_state = None
        self.current_mode = None

    def _setup_device(self):
        torch.autocast(device_type=self.device, dtype=torch.bfloat16). __enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        print("Cuda available:", torch.cuda.is_available())
        torch.cuda.set_device(0)
        print("Using device:", torch.cuda.get_device_name(0))

    def _get_frame_names(self):
        frame_names = [
            p for p in os.listdir(self.video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        return frame_names

    def show_frame(self, frame_idx=0):
        plt.figure(figsize=(12, 8))
        plt.title(f"frame {frame_idx}")
        plt.imshow(Image.open(os.path.join(self.video_dir, self.frame_names[frame_idx])))
        plt.show()

    def set_image(self, image_path: str):
        image = Image.open(image_path)
        self.image_predictor.set_image(image)
        self.current_mode = 'image'

    def initialize_inference(self, mode='video'):
        if mode == 'video':
            self.video_state = self.video_predictor.init_state(video_path=self.video_dir)
            self.video_predictor.reset_state(self.video_state)
        elif mode == 'image':
            pass
        self.current_mode = mode

    def annotate_frame(self, ann_frame_idx, ann_obj_id, points, labels):
        if self.current_mode == 'video':
            print(f"Debug: Annotating frame {ann_frame_idx} with points {points} and labels {labels}")
            _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points(
                inference_state=self.video_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )
            self._show_annotation(ann_frame_idx, points, labels, out_mask_logits, out_obj_ids)
        elif self.current_mode == 'image':
            out_mask, _, _ = self.image_predictor.predict(
                point_coords=points,
                point_labels=labels
            )
            self._show_annotation(ann_frame_idx, points, labels, out_mask, [ann_obj_id])

    def _show_annotation(self, frame_idx, points, labels, out_mask_logits, out_obj_ids):
        plt.figure(figsize=(12, 8))
        plt.title(f"frame {frame_idx}")
        if self.current_mode == 'video':
            plt.imshow(Image.open(os.path.join(self.video_dir, self.frame_names[frame_idx])))
        elif self.current_mode == 'image':
            plt.imshow(Image.open(self.video_dir))

        self.show_points(points, labels, plt.gca())
        self.show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        plt.show()

    @staticmethod
    def show_mask(mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])

        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @staticmethod
    def show_points(coords, labels, ax, marker_size=200):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25)

    def propagate_and_collect_results(self):
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(self.video_state):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    def render_segmentation_results(self, vis_frame_stride=1):
        plt.close("all")
        fig = plt.figure(figsize=(6, 4))
        for out_frame_idx in range(0, len(self.frame_names), vis_frame_stride):
            plt.title(f"frame {out_frame_idx}")
            im = plt.imshow(Image.open(os.path.join(self.video_dir, self.frame_names[out_frame_idx])), animated=True)
            for out_obj_id, out_mask in self.video_segments[out_frame_idx].items():
                self.show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
                plt.savefig(f"outputs/seg_frames/s{out_frame_idx}.png")

    def predict_image(self, points, labels):
        out_mask, _, _ = self.image_predictor.predict(
            point_coords=points,
            point_labels=labels
        )
        self._show_annotation(0, points, labels, out_mask, [1])

def main():
    parser = argparse.ArgumentParser(description="Run SAM2 video processing")
    parser.add_argument("--video_dir", type=str, required=True, help="Path to the directory containing video frames or image file")
    parser.add_argument("--model_cfg", type=str, required=True, help="Model configuration file")
    parser.add_argument("--sam2_checkpoint", type=str, required=True, help="Path to the SAM2 model checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output video")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video")
    parser.add_argument("--quality", type=int, default=10, help="Quality of the output video")
    parser.add_argument("--max_frames", type=int, default=250, help="Maximum number of frames to process")
    parser.add_argument("--scale", type=str, default="640:-1", help="Scale for the output video")
    parser.add_argument("--obj_id", type=int, required=True, help="Object ID for annotation")
    parser.add_argument("--mode", type=str, choices=['image', 'video'], required=True, help="Processing mode: 'image' or 'video'")
    parser.add_argument("--image_path", type=str, help="Path to the image file for image mode")

    args = parser.parse_args()

    sam2_model = SAM2Model(args.video_dir, args.model_cfg, args.sam2_checkpoint)

    if args.mode == 'image' and args.image_path:
        sam2_model.set_image(args.image_path)

    sam2_model.show_frame(0)
    print("Press enter to continue...")
    input()
    print("Continuing...")

    sam2_model.initialize_inference(mode=args.mode)
    
    ann_frame_idx = 0
    ann_obj_id = args.obj_id
    points = np.array([[274, 189], [323, 161]], dtype=np.float32)
    labels = np.array([1, 1], dtype=np.int32)
    
    if args.mode == 'image':
        sam2_model.predict_image(points, labels)
    else:
        sam2_model.annotate_frame(ann_frame_idx, ann_obj_id, points, labels)

    print("Press enter to continue...")
    input()
    print("Continuing...")

    if args.mode == 'video':
        sam2_model.propagate_and_collect_results()
        sam2_model.render_segmentation_results()

if __name__ == "__main__":
    main()
