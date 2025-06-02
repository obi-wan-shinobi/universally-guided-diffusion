from ugdiff.models.diffusion import generate_image

if __name__ == "__main__":
    prompt = "Spartan"
    generate_image(prompt, "tmp/sample.png")
