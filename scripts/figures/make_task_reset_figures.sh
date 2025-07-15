# tasks w/ viz of reset plane
python scripts/figures/make_video_resets.py --task lift-narrow --frames 1 --viz-reset-plane
python scripts/figures/make_video_resets.py --task lift-wide --frames 1 --viz-reset-plane
python scripts/figures/make_video_resets.py --task can-narrow --frames 1 --viz-reset-plane
python scripts/figures/make_video_resets.py --task can-wide --frames 1 --viz-reset-plane
python scripts/figures/make_video_resets.py --task square-narrow --frames 1 --viz-reset-plane
python scripts/figures/make_video_resets.py --task square-wide --frames 1 --viz-reset-plane

# then, for showing individual objects and them being augmented, use the crop-img.py script
