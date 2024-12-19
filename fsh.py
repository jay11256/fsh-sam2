import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from PIL import Image

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    # device = torch.device("cpu")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

# if device.type == "cuda":
#     # use bfloat16 for the entire notebook
#     torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     if torch.cuda.get_device_properties(0).major >= 8:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
# elif device.type == "mps":
#     print(
#         "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
#         "give numerically different outputs and sometimes degraded performance on MPS. "
#         "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
#     )

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "../checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

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
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "./ultrashort_jpg"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# Displays specific frame number
def display_frame(frame_idx):
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
    plt.show()

inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)

prompts = {}

'''
Create a function to make adding clicks easier
Params:
- frame_idx: int
- obj_id: int
- points_arr: [int, int] array
- labels_arr: int array
- display: bool
'''
def add_point(frame, obj, points_arr, labels_arr, display):
    points = np.array(points_arr, dtype=np.float32)
    labels = np.array(labels_arr, np.int32)
    # prompts[obj] = points, labels
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state = inference_state,
        frame_idx = frame,
        obj_id = obj,
        points = points,
        labels = labels,
    )

    if display:
        plt.figure(figsize = (9, 6))
        plt.title(f"frame {frame}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame])))
        show_points(points, labels, plt.gca())
        for i, out_obj_id in enumerate(out_obj_ids):
            # show_points(*prompts[out_obj_id], plt.gca())
            show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id = out_obj_id)
        plt.show()

def propagate(vis_frame_stride, display):
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # render the segmentation results every few frames
    plt.close("all")
    if display:
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            plt.figure(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            plt.show()

    return video_segments

if __name__ == "__main__":
    # display_frame(0)
    disp = False
    add_point(0, 1, [[964, 669], [1050, 640]], [1, 1], disp)
    add_point(0, 2, [[1077, 561]], [1], disp)
    vid_seg = propagate(3, True)

    # for key, value in vid_seg.items():
    #     for vkey, vvalue in value.items():
    #         print(f"Key: {key}, VKey: {vkey}, VValue: {type(vvalue)}")

    # # Convert NumPy arrays to lists for JSON serialization
    # serializable_data = {
    #     outer_key: {
    #         inner_key: inner_value.tolist() for inner_key, inner_value in outer_value.items()
    #     }
    #     for outer_key, outer_value in vid_seg.items()
    # }

    # # Save to JSON file
    # with open("data.json", "w") as f:
    #     json.dump(serializable_data, f)