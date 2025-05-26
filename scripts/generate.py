from ugdiff.models.diffusion import generate_image

if __name__ == "__main__":
    prompt = "A Studio Ghibli calming scenery painting with Chihiro eating."
    generate_image(prompt, "tmp/sample.png")
