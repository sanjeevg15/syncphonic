import gensim.downloader as api
from gensim.models import KeyedVectors
from run_placesCNN_basic import get_image_tags

#from visualize import display_pca_scatterplot

# if __name__ == "__main__":

def similar(img):
    info = api.info()
    model = api.load("word2vec-google-news-300")
    #model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300-SLIM.bin.gz', binary=True)

    # moods = ['Uplifting',
    #          'Epic',
    #          'Powerful',
    #          'Exciting',
    #          'Happy',
    #          'Funny',
    #          'Carefree',
    #          'Hopeful',
    #          'Love',
    #          'Playful',
    #          'Groovy',
    #          'Sexy',
    #          'Peaceful',
    #          'Mysterious',
    #          'Serious',
    #          'Dramatic',
    #          'Angry',
    #          'Tense',
    #          'Sad', ]

    scenes = ['Travel',
            'Nature',
            'Sports',
            'Food',
            'City']

    scene_info = []

    base_words = get_image_tags(img)

    #print("scene")

    for i in base_words:
        scene_info.append(i[1])

    mood_similarities = [0] * len(scenes)

    for weight, base_word in base_words:
        for i, mood in enumerate(scenes):
            try:
                base_word = base_word.split('_')[0]
                base_word = base_word.split('/')[0]
                cur_similarity = model.similarity(base_word, mood.lower())
    
                #print(cur_similarity)
                mood_similarities[i] += weight * cur_similarity
                #print("Similarity between {} and {}".format(base_word, mood.lower()), cur_similarity)
            except:
                continue

    mood_similarities = [x / len(base_words) for x in mood_similarities]

    sorted_moods = [y for x, y in sorted(zip(mood_similarities, scenes), key=lambda x: x[0])]
    sorted_moods.reverse()

    # print(list(zip(scenes, mood_similarities)))
    # print(sorted_moods)

    sorted_moods = [x.lower() for x in sorted_moods]

    #print(sorted_moods[0])

    final_mood = sorted_moods[0]

    #print(scene_info)

    return scene_info,  final_mood


# scene_info = similar()


