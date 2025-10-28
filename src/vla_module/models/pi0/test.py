from vla_module.models.pi0.PI0Policy import PI0Policy


def main() -> None:
	model = PI0Policy.load_stored_weights("model.safetensors")
	print("Loaded PI0 policy:", type(model))


if __name__ == "__main__":
	main()