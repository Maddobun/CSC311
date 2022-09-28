import models.hm1_code as dtree

if __name__ == "__main__":
    train, val, test = dtree.load_data()
    dtree.select_model(train, val)
