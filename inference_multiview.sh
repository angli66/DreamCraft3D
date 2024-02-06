wandb_project="dreamcraft3d-multiview-perturb"

if [ "$1" == "factory" ]; then
    name="factory"
    prompt="brick factory with smoking chimneys and a character by the door"
    images_path="load/multiview_images/factory"
    image_path="load/multiview_images/factory/0_rgba.png"
elif [ "$1" == "shiba" ]; then
    name="shiba"
    prompt="a shiba inu dog with a red collar"
    images_path="load/multiview_images/shiba"
    image_path="load/multiview_images/shiba/0_rgba.png"
elif [ "$1" == "laptop" ]; then
    name="laptop"
    prompt="laptop with a cyberpunk city wallpaper"
    images_path="load/multiview_images/laptop"
    image_path="load/multiview_images/laptop/0_rgba.png"
elif [ "$1" == "character" ]; then
    name="character"
    prompt="female character in an assassin costume with crossbows"
    images_path="load/multiview_images/character"
    image_path="load/multiview_images/character/0_rgba.png"
else
    echo "Supported choices are factory, shiba, laptop and character"
fi

# --------- Stage 1 (NeRF & NeuS) --------- #
python launch.py --config configs/dreamcraft3d-coarse-nerf-multiview.yaml --train \
    name="$name-nerf" \
    system.prompt_processor.prompt="$prompt" \
    data.image_path="$image_path" \
    data.images_path="$images_path" \
    system.loggers.wandb.project="$wandb_project"

ckpt=outputs_multiview/$name-nerf/$prompt@LAST/ckpts/last.ckpt
python launch.py --config configs/dreamcraft3d-coarse-neus-multiview.yaml --train \
    name="$name-neus" \
    system.prompt_processor.prompt="$prompt" \
    data.image_path="$image_path" \
    data.images_path="$images_path" \
    system.weights="$ckpt" \
    system.loggers.wandb.project="$wandb_project"

# --------- Stage 2 (Geometry Refinement) --------- #
ckpt=outputs_multiview/$name-neus/$prompt@LAST/ckpts/last.ckpt
python launch.py --config configs/dreamcraft3d-geometry-multiview.yaml --train \
    name="$name-dmtet" \
    system.prompt_processor.prompt="$prompt" \
    data.image_path="$image_path" \
    data.images_path="$images_path" \
    system.geometry_convert_from="$ckpt" \
    system.loggers.wandb.project="$wandb_project"

# --------- Stage 3 (Texture Refinement) --------- #
ckpt=outputs_multiview/$name-dmtet/$prompt@LAST/ckpts/last.ckpt
python launch.py --config configs/dreamcraft3d-texture-multiview.yaml --train \
    name="$name-texture" \
    system.prompt_processor.prompt="$prompt" \
    data.image_path="$image_path" \
    data.images_path="$images_path" \
    system.geometry_convert_from="$ckpt" \
    system.loggers.wandb.project="$wandb_project"
