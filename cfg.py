class CFG:
    class_nums = 1
    ckpt_path = "./save_model/20231015/"
    data_path = "./data/"
    model_name = "unet"

    n_fold = 5
    image_size = 224
    train_bs = 4
    valid_bs = train_bs * 2

    epoch = 5
    lr = 0.001
    weight_decay = 0.0001
    seed = 42
    thr = 0.3
    resume = False
