from data import ReviewDataset
from nbc import NBC
import matplotlib.pyplot as plt

DICT_COUNT = 5000
PATH = "./yelp_academic_dataset_review/yelp_academic_dataset_review.json"
LABEL_NAMES = ['isFunny', 'isUseful', 'isCool', 'isPositive']
N_TRAIN = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
N_TEST = 8000


def zero_one_loss(pred, actual):
    return 1 / len(pred) * sum(0 if pred[i] == actual[i] else 1 for i in range(len(pred)))


def main():
    n_total = sum(N_TRAIN) + N_TEST

    data = ReviewDataset(PATH, DICT_COUNT)
    loss = []
    for i, label_name in enumerate(LABEL_NAMES):
        loss.append([])
        features, labels = data.get_bags(label_name, n_total)
        test_f, test_l = features[n_total-N_TEST:], labels[n_total-N_TEST:]

        for n_train in N_TRAIN:
            train_f, train_l = features[:n_train], labels[:n_train]
            features, labels = features[n_train:], labels[n_train:]
            nbc = NBC(train_f, train_l)
            predictions = nbc.classify(test_f)
            loss[i].append(zero_one_loss(predictions, test_l))

        # plot graph
        plt.title(label_name)
        plt.plot(N_TRAIN, loss[i])
        plt.xlabel("Training set size")
        plt.ylabel("Zero-one loss")
        plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
