def load_pretrained_model(model_dir):
    """
    Loads the pretrained model on the model_dir.
    
    Parameter:
    model_dir(String) - path to the pretrained model.

    Returns:
    pretrained Model object.
    """
    try:
        try:
            try:
                from model.RIFE_HDv2 import Model
                model = Model()
                model.load_model(model_dir, -1)
                print("Loaded v2.x HD model.")
            except:
                from train_log.RIFE_HDv3 import Model
                model = Model()
                model.load_model(model_dir, -1)
                print("Loaded v3.x HD model.")
        except:
            from model.RIFE_HD import Model
            model = Model()
            model.load_model(model_dir, -1)
            print("Loaded v1.x HD model")
    except:
        from model.RIFE import Model
        model = Model()
        model.load_model(model_dir, -1)
        print("Loaded ArXiv-RIFE model")
    model.eval()
    model.device()
    return model