def gan_args(parser):
    parser.add_argument(
        "--bs", default=4, type=int, help="Batch Size (default: 1)",
    )
    parser.add_argument(
        "--n_channels", default=3, type=int, help="Batch Size (default: 1)",
    )
    parser.add_argument(
        "--dim_z_content", default=50, type=int, help="Batch Size (default: 1)",
    )
    parser.add_argument(
        "--dim_z_category", default=6, type=int, help="Batch Size (default: 1)",
    )
    parser.add_argument(
        "--dim_z_motion", default=10, type=int, help="Batch Size (default: 1)",
    )
    parser.add_argument(
        "--video_length", default=8, type=int, help="Batch Size (default: 1)",
    )
    parser.add_argument(
        "--num_labels", default=400, type=int, help="Batch Size (default: 1)",
    )
    parser.add_argument(
        "--lr_c", default=1e-4, type=float, help="Batch Size (default: 1)",
    )
    parser.add_argument(
        "--lr_g", default=1e-4, type=float, help="Batch Size (default: 1)",
    )
    parser.add_argument(
        "--num_epochs", default=50, type=int, help="Batch Size (default: 1)",
    )
    parser.add_argument(
        "--alpha", default=0.1, type=float, help="Batch Size (default: 1)",
    )