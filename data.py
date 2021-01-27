import json
from tqdm import tqdm
import re
import random


class Review:
    def __init__(self, votes, user_id, review_id, stars, date, text, rev_type, business_id):
        self.votes = votes
        self.user_id = user_id
        self.review_id = review_id
        self.stars = stars
        self.date = date
        self.text = text
        self.rev_type = rev_type
        self.business_id = business_id


def review_decoder(obj):
    if obj.get('votes'):
        return Review(obj['votes'], obj['user_id'], obj['review_id'], obj['stars'],
                      obj['date'], obj['text'], obj['type'], obj['business_id'])
    else:
        return obj


def to_wordlist(sentence):
    return re.sub('[^a-zA-Z]', ' ', sentence).lower().split()


def get_num_lines(path):
    with open(path, "r+") as file:
        return sum(1 for _ in file)


class ReviewDataset:
    def __init__(self, path, dict_count):
        self.path = path
        self.dict_count = dict_count
        self.__load_reviews__()
        self.__create_wordlist__(dict_count)

    def __load_reviews__(self):
        self.reviews = []
        with open(self.path) as file:
            reviews_added = 0
            for line in tqdm(file, total=get_num_lines(self.path), desc='Load reviews: '):
                review = json.loads(line, object_hook=review_decoder)
                if 2 < sum(review.votes.values()) < 11:
                    self.reviews.append(review)
                    reviews_added += 1
            print(f'Added {reviews_added} reviews to the dataset.')

    def __create_wordlist__(self, word_count):
        word_dict = {}
        for review in tqdm(self.reviews, total=len(self.reviews), desc='Create word_list: '):
            words = to_wordlist(review.text)
            for word in words:
                word_dict[word] = word_dict.get(word, 0) + 1
        words_sorted = sorted(word_dict, key=word_dict.get, reverse=True)
        self.word_list = words_sorted[:word_count]
        print(f'Created word_list with the {word_count} most frequent words.')

    def get_bags(self, label, n_bags):
        int_labels = []
        features = []
        for review in tqdm(random.sample(self.reviews, n_bags), total=n_bags,
                           desc='Create ' + str(n_bags) + ' bags for label ' + label + ': '):
            word_dict = dict.fromkeys(self.word_list, 0)
            for word in to_wordlist(review.text):
                if word in self.word_list:
                    word_dict[word] += 1

            # Create labels for each review
            labels = ['is' + str(word).capitalize() for word, count in review.votes.items() if count > 0]
            if review.stars > 3.5:
                labels.append('isPositive')

            int_labels.append(1 if label in labels else 0)
            features.append(list(word_dict.values()))

        return features, int_labels


if __name__ == '__main__':
    path = "./yelp_academic_dataset_review/yelp_academic_dataset_review.json"
    data = ReviewDataset(path, 500)
    print(data.reviews.pop(0).text)
    print(len(data.word_list))
    # print(data.word_list)
    train, test = data.get_bags('isFunny', 30)
    print(train[0][0])
    print(train[1])
    print(test[1])
