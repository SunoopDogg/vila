#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from termcolor import colored

import llava
from llava import conversation as clib
from llava.media import Image
from llava.utils.logging import logger


def find_images(images_dir: str) -> List[Path]:
    """Find all supported image files in the images directory."""
    supported_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    image_files = []

    images_path = Path(images_dir)
    if not images_path.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return []

    for ext in supported_extensions:
        image_files.extend(images_path.glob(f"*{ext}"))
        image_files.extend(images_path.glob(f"*{ext.upper()}"))

    return sorted(image_files)


def display_images(image_files: List[Path]) -> None:
    """Display available images with numbered selection."""
    print(colored("\n=== Available Images ===", "cyan", attrs=["bold"]))
    for idx, img_path in enumerate(image_files, 1):
        file_size = img_path.stat().st_size / 1024  # KB
        print(f"  [{idx}] {img_path.name} ({file_size:.1f} KB)")
    print()


def select_images(image_files: List[Path]) -> List[Path]:
    """Prompt user to select images by number or name."""
    while True:
        selection = input(
            colored("Select image(s) [number or 'all']: ", "yellow")).strip()

        if not selection:
            print(colored("Please enter a selection.", "red"))
            continue

        if selection.lower() == "all":
            return image_files

        try:
            # Support comma-separated numbers
            indices = [int(idx.strip()) for idx in selection.split(",")]
            selected = []
            for idx in indices:
                if 1 <= idx <= len(image_files):
                    selected.append(image_files[idx - 1])
                else:
                    print(colored(f"Invalid index: {idx}", "red"))
                    return []

            if selected:
                return selected
        except ValueError:
            # Try to find by filename
            matches = [img for img in image_files if selection.lower()
                       in img.name.lower()]
            if matches:
                return matches
            print(
                colored(f"Could not find image matching: {selection}", "red"))


def get_text_input() -> Optional[str]:
    """Get text query from user."""
    print(colored("\nEnter your question (or 'quit' to exit):", "yellow"))
    text = input(colored("> ", "green")).strip()

    if text.lower() in ["quit", "exit", "q"]:
        return None

    if not text:
        print(colored("Please enter a question.", "red"))
        return ""

    return text


def configure_ps3_and_context_length(model):
    """Configure PS3 settings and adjust context length based on environment variables."""
    # Get PS3 configs from environment variables
    num_look_close = os.environ.get("NUM_LOOK_CLOSE", None)
    num_token_look_close = os.environ.get("NUM_TOKEN_LOOK_CLOSE", None)
    select_num_each_scale = os.environ.get("SELECT_NUM_EACH_SCALE", None)
    look_close_mode = os.environ.get("LOOK_CLOSE_MODE", None)
    smooth_selection_prob = os.environ.get("SMOOTH_SELECTION_PROB", None)

    # Set PS3 configs
    if num_look_close is not None:
        logger.info(f"Num look close: {num_look_close}")
        model.num_look_close = int(num_look_close)
    if num_token_look_close is not None:
        logger.info(f"Num token look close: {num_token_look_close}")
        model.num_token_look_close = int(num_token_look_close)
    if select_num_each_scale is not None:
        logger.info(f"Select num each scale: {select_num_each_scale}")
        select_num_each_scale = [int(x)
                                 for x in select_num_each_scale.split("+")]
        model.get_vision_tower(
        ).vision_tower.vision_model.max_select_num_each_scale = select_num_each_scale
    if look_close_mode is not None:
        logger.info(f"Look close mode: {look_close_mode}")
        model.look_close_mode = look_close_mode
    if smooth_selection_prob is not None:
        logger.info(f"Smooth selection prob: {smooth_selection_prob}")
        if smooth_selection_prob.lower() == "true":
            smooth_selection_prob = True
        elif smooth_selection_prob.lower() == "false":
            smooth_selection_prob = False
        else:
            raise ValueError(
                f"Invalid smooth selection prob: {smooth_selection_prob}")
        model.smooth_selection_prob = smooth_selection_prob

    # Adjust the max context length based on the PS3 config
    context_length = model.tokenizer.model_max_length
    if num_look_close is not None:
        context_length = max(context_length, int(
            num_look_close) * 2560 // 4 + 1024)
    if num_token_look_close is not None:
        context_length = max(context_length, int(
            num_token_look_close) // 4 + 1024)
    context_length = max(
        getattr(model.tokenizer, "model_max_length", context_length), context_length)
    model.config.model_max_length = context_length
    model.config.tokenizer_model_max_length = context_length
    model.llm.config.model_max_length = context_length
    model.llm.config.tokenizer_model_max_length = context_length
    model.tokenizer.model_max_length = context_length


def main() -> None:
    parser = argparse.ArgumentParser(description="Local VILA Image Analyzer")
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default="NVILA-Lite-8B",
        help="Path to the VILA model"
    )
    parser.add_argument(
        "--images-dir",
        "-i",
        type=str,
        default="images",
        help="Directory containing images"
    )
    parser.add_argument(
        "--conv-mode",
        "-c",
        type=str,
        default="vicuna_v1",
        help="Conversation mode (vicuna_v1, llama_3, etc.)"
    )
    parser.add_argument(
        "--lora-path",
        "-l",
        type=str,
        default=None,
        help="Optional LoRA weights path"
    )
    args = parser.parse_args()

    # Print banner
    print(colored("\n" + "=" * 60, "cyan", attrs=["bold"]))
    print(colored("  VILA Local Image Analyzer", "cyan", attrs=["bold"]))
    print(colored("=" * 60 + "\n", "cyan", attrs=["bold"]))

    # Load model
    logger.info(f"Loading model from: {args.model_path}")
    if args.lora_path is None:
        model = llava.load(args.model_path, model_base=None)
    else:
        model = llava.load(args.lora_path, model_base=args.model_path)

    logger.info("Model loaded successfully")

    # Configure PS3 and context length
    configure_ps3_and_context_length(model)

    # Set conversation mode
    clib.default_conversation = clib.conv_templates[args.conv_mode].copy()
    logger.info(f"Using conversation mode: {args.conv_mode}")

    # Find available images
    image_files = find_images(args.images_dir)

    if not image_files:
        logger.error(f"No images found in {args.images_dir}")
        logger.info(
            "Please add some images (.jpg, .jpeg, .png) to the images/ folder")
        sys.exit(1)

    logger.info(f"Found {len(image_files)} image(s) in {args.images_dir}")

    # Interactive loop
    print(colored("\nInteractive mode started. Type 'quit' to exit.\n", "green"))

    while True:
        try:
            # Display and select images
            display_images(image_files)
            selected_images = select_images(image_files)

            if not selected_images:
                continue

            print(
                colored(f"\nSelected {len(selected_images)} image(s):", "green"))
            for img in selected_images:
                print(f"  - {img.name}")

            # Get text input
            text = get_text_input()

            if text is None:  # User wants to quit
                print(colored("\nExiting...", "yellow"))
                break

            if not text:  # Empty input
                continue

            # Prepare prompt
            prompt = []
            for img_path in selected_images:
                prompt.append(Image(str(img_path)))
            prompt.append(text)

            # Generate response
            print(colored("\n[Processing...]", "yellow"))
            response = model.generate_content(prompt)

            # Display response
            print(colored("\n=== Response ===", "cyan", attrs=["bold"]))
            print(colored(response, "white", attrs=["bold"]))
            print(colored("=" * 60 + "\n", "cyan"))

        except KeyboardInterrupt:
            print(colored("\n\nInterrupted. Exiting...", "yellow"))
            break
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            print(colored(f"\nError: {e}", "red"))
            continue


if __name__ == "__main__":
    main()
