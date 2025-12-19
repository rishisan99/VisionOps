def main():
    from src.components.data_loader import create_ssl_dataloader

    loader = create_ssl_dataloader(
        data_dir="Artifacts/12_18_2025_23_15_29/data_splitting/processed_split/train",
        batch_size=8,
        num_workers=2
    )

    x1, x2 = next(iter(loader))
    print(x1.shape, x2.shape)


if __name__ == "__main__":
    main()
