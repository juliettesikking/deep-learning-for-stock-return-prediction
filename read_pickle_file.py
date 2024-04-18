objects = []
with (open("/Users/juliette/Documents/bachelor_projet_deep_learning/projet/shallow_net_model.pth", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break