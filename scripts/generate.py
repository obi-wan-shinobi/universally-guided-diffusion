from ugdiff.models.diffusion import generate_image

if __name__ == "__main__":
    prompt = "TU Delft EEMCS building in Van Gogh's starry night style"
    generate_image(prompt, "tmp/sample.png")
