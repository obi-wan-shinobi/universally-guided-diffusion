from ugdiff.models.diffusion import generate_image

if __name__ == "__main__":
    prompt = "a dog in space"
    generate_image(prompt, "tmp/sample.png")
