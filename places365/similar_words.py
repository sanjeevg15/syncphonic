import gensim.downloader as api
from run_placesCNN_basic import get_image_tags

if __name__ == "__main__":

    info = api.info()
    model = api.load("word2vec-google-news-300")

    moods = ['Uplifting',
             'Epic',
             'Powerful',
             'Exciting',
             'Happy',
             'Funny',
             'Carefree',
             'Hopeful',
             'Love',
             'Playful',
             'Groovy',
             'Sexy',
             'Peaceful',
             'Mysterious',
             'Serious',
             'Dramatic',
             'Angry',
             'Tense',
             'Sad', ]

    base_words = get_image_tags('12.jpg')

    print(base_words)

    mood_similarities = [0] * len(moods)

    for weight, base_word in base_words:
        for i, mood in enumerate(moods):
            try:
                cur_similarity = model.similarity(base_word, mood.lower())
                mood_similarities[i] += weight * cur_similarity
                print("Similarity between {} and {}".format(base_word, mood.lower()), cur_similarity)
            except:
                continue

    mood_similarities = [x / len(base_words) for x in mood_similarities]

    sorted_moods = [y for x, y in sorted(zip(mood_similarities, moods), key=lambda x: x[0])]
    sorted_moods.reverse()

    print(list(zip(moods, mood_similarities)))
    print(sorted_moods)

    sorted_moods = [x.lower() for x in sorted_moods]

