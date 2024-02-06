wandb_project="dreamcraft3d-singleview"

if [ "$1" == "factory" ]; then
    name="factory"
    prompt="brick factory with smoking chimneys and a character by the door"
    image_path="load/multiview_images/factory/0_rgba.png"
elif [ "$1" == "shiba" ]; then
    name="shiba"
    prompt="a shiba inu dog with a red collar"
    image_path="load/multiview_images/shiba/0_rgba.png"
elif [ "$1" == "laptop" ]; then
    name="laptop"
    prompt="laptop with a cyberpunk city wallpaper"
    image_path="load/multiview_images/laptop/0_rgba.png"
elif [ "$1" == "character" ]; then
    name="character"
    prompt="female character in an assassin costume with crossbows"
    image_path="load/multiview_images/character/0_rgba.png"
else
    echo "Supported choices are factory, shiba, laptop and character"
fi

# --------- Stage 1 (NeRF & NeuS) --------- #
python launch.py --config configs/dreamcraft3d-coarse-nerf.yaml --train \
    name="$name-nerf" \
    system.prompt_processor.prompt="$prompt" \
    data.image_path="$image_path" \
    system.loggers.wandb.project="$wandb_project"

ckpt=outputs/$name-nerf/$prompt@LAST/ckpts/last.ckpt
python launch.py --config configs/dreamcraft3d-coarse-neus.yaml --train \
    name="$name-neus" \
    system.prompt_processor.prompt="$prompt" \
    data.image_path="$image_path" \
    system.weights="$ckpt" \
    system.loggers.wandb.project="$wandb_project"

# --------- Stage 2 (Geometry Refinement) --------- #
ckpt=outputs/$name-neus/$prompt@LAST/ckpts/last.ckpt
python launch.py --config configs/dreamcraft3d-geometry.yaml --train \
    name="$name-dmtet" \
    system.prompt_processor.prompt="$prompt" \
    data.image_path="$image_path" \
    system.geometry_convert_from="$ckpt" \
    system.loggers.wandb.project="$wandb_project"

# --------- Stage 3 (Texture Refinement) --------- #
ckpt=outputs/$name-dmtet/$prompt@LAST/ckpts/last.ckpt
python launch.py --config configs/dreamcraft3d-texture.yaml --train \
    name="$name-texture" \
    system.prompt_processor.prompt="$prompt" \
    data.image_path="$image_path" \
    system.geometry_convert_from="$ckpt" \
    system.loggers.wandb.project="$wandb_project"
