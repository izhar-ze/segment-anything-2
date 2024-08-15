# To extract frames from a video:
# ffmpeg -i video.mp4 -q:v 2 -start_number 0 frames/%05d.jpg

# Lower quality and fewer frames
# ffmpeg -i video.mp4 -vf "scale=640:-1" -q:v 10 -vframes 250 -start_number 0 fox_frames/%05d.jpg

# To create a video from frames:
# ffmpeg -framerate 30 -i outputs/seg_frames/s%d.png -c:v libx264 -r 30 outputs/seg_video.mp4

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

# use bfloat16 for faster inference
torch.autocast(device_type="cuda", dtype=torch.bfloat16). __enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print("Cuda available:", torch.cuda.is_available())
torch.cuda.set_device(0)
torch.cuda.current_device()
torch.cuda.get_device_name(0)


sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

# # mask function
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

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25)


video_dir = "videos/fox_frames" # the directory containing the video frames

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


# take a look at the first video frame (or change), and find the object points at least two of them
frame_idx = 0
plt.figure(figsize=(12, 8))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
plt.show()

print("Press enter to continue...")
input()
print("Continuing...")

inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)

ann_frame_idx = 0   # the frame index we interact with
ann_obj_id = 1      # give a unique id to each object we interact with (it can be any integers)

# here enter the point coordinates [x, y] as rows of a numpy array
# points = np.array([[274, 189], [286, 169], [323, 161], [337, 140]], dtype=np.float32)
points = np.array([[274, 189], [323, 161]], dtype=np.float32)

# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1], dtype=np.int32)

_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))

show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
plt.show()

print("Press enter to continue...")
input()
print("Continuing...")

# run propagation throughout the video and collect the results in a dict
video_segments = {} # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 1
plt.close("all")

# defind the figure outside the loop
fig = plt.figure(figsize=(6, 4))
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.title(f"frame {out_frame_idx}")
    im = plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])), animated=True)
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

        # here, just save the files with an increasing index in the folder called output
        plt.savefig(f"outputs/seg_frames/s{out_frame_idx}.png")