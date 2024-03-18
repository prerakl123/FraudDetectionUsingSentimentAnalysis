import warnings

from transformers import pipeline
import whisper

from audio_text_extraction import get_transcript_from_chunks

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")

model = whisper.load_model("base")
sentiment_analysis_michelle = pipeline(
    task="sentiment-analysis",
    framework="pt",
    model="michellejieli/emotion_text_classifier",
    device="cuda",
    max_length=512,
    truncation=True
)
sentiment_analysis_samlowe = pipeline(
    task="text-classification",
    framework="pt",
    model="SamLowe/roberta-base-go_emotions",
    device="cuda",
    max_length=512,
    truncation=True,
    top_k=None
)
analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment_michelle(text):
    results = sentiment_analysis_michelle(text)
    print(results)
    sentiment_results = {
        result['label']: result['score'] for result in results
    }
    return sentiment_results


def analyze_sentiment_samlowe(text):
    results = sentiment_analysis_samlowe(text)
    print(results)
    sentiment_results = {
        result['label']: result['score'] for result in results[0]
    }
    return sentiment_results


def display_sentiment_results(sentiment_results):
    sentiment_text = ""
    for sentiment, score in sentiment_results.items():
        sentiment_text += f"{sentiment} : {score}\n"
    return sentiment_text


# print(display_sentiment_results(
#     analyze_sentiment(
#         get_transcript_from_chunks(
#             "audio/video_1_mp4_chunked/subtitles.txt"
#         )
#     )
# ))


def analyze_sentiment_from_vader(text):
    # Pre-process Text
    tokens = word_tokenize(text.lower())

    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return analyzer.polarity_scores(processed_text)


sentence_list = [
    # JOY
    "The children laughed and played in the sun-dappled meadow, their joy infectious. Each giggle echoed the pure "
    "delight of being alive, filling the air with warmth.",

    "Amidst a sea of smiling faces, friends gathered around a picnic blanket, sharing stories and laughter. The air "
    "buzzed with the sound of joyous chatter as they savored the simple pleasures of companionship and camaraderie. "
    "Moments like these, filled with genuine connection, were cherished reminders of life's abundant blessings.",

    "In a bustling city park, a vibrant festival unfolded, drawing people from all walks of life. Colorful banners "
    "fluttered in the breeze, and music echoed off the surrounding skyscrapers. Families picnicked on lush green "
    "lawns, while performers captivated audiences with their talents. Everywhere, there was laughter and merriment, "
    "as strangers became friends in the spirit of celebration. Joy permeated the atmosphere, binding hearts together "
    "in a shared experience of happiness and gratitude. It was a day to remember, a testament to the enduring power "
    "of community and togetherness in spreading joy throughout the world.",


    # SADNESS
    "Tears welled in her eyes as memories flooded back, a bittersweet reminder of what once was. The ache in her "
    "chest seemed to grow heavier with each passing moment.",

    "In the solitude of her dimly lit room, she sat with a heavy heart, surrounded by the echoes of a love lost. "
    "Photographs on the wall whispered of happier times, now distant and unattainable. The weight of loneliness "
    "settled upon her shoulders like a heavy cloak, suffocating in its intensity.",

    "Underneath a veil of gray clouds, the world seemed to mourn alongside her, mirroring the heaviness of her soul. "
    "Rain pattered against the windowpanes, a somber melody that matched the rhythm of her tears. In the stillness of "
    "the night, she clung to the fragments of a shattered dream, grappling with the pain of letting go. Grief washed "
    "over her in waves, threatening to pull her under, but amidst the darkness, a flicker of hope remained—a reminder "
    "that even in the depths of despair, there was the promise of healing and renewal.",


    # ANGER
    "His fists clenched at his sides, knuckles turning white with the force of his suppressed rage. Each breath felt "
    "like fire in his lungs as he struggled to contain the storm raging within.",

    "Voices rose in a heated exchange, words spat like venom, leaving wounds that cut deeper than any physical blow. "
    "The air crackled with tension, thick with the weight of unspoken grievances. Anger boiled just beneath the "
    "surface, threatening to erupt like a volcano on the verge of eruption.",

    "In the aftermath of a heated argument, the silence was deafening, punctuated only by the sound of heavy "
    "breathing and pounding hearts. Furniture lay overturned, a testament to the violence of unchecked emotions. "
    "Regret hung in the air like a dense fog, suffocating in its intensity. But amidst the wreckage, there was a "
    "glimmer of realization—a recognition of the destructive power of anger and the need for forgiveness and "
    "reconciliation to rebuild what had been broken.",


    # FEAR
    "Her heart raced, pounding in her chest like a drumbeat of impending doom. Every shadow seemed to leer "
    "menacingly, whispering of unseen threats lurking in the darkness.",

    "As night fell, she found herself alone in the vast expanse of the forest, the trees looming like silent "
    "sentinels in the gloom. Every rustle of leaves sent a shiver down her spine, her imagination running wild with "
    "imagined terrors lurking just beyond her line of sight. Fear gripped her like icy fingers, paralyzing her with "
    "its suffocating grip.",

    "In the depths of the abandoned mansion, she ventured forth with trepidation, the creaking floorboards echoing "
    "her racing heartbeat. Shadows danced along the walls, twisting and contorting into grotesque shapes that seemed "
    "to taunt her with their malevolence. Each step forward felt like a leap into the unknown, fraught with the "
    "potential for unseen dangers lurking in the darkness. But amidst the terror, there was a stubborn "
    "determination—a refusal to succumb to fear and a resolve to confront whatever horrors awaited her in the shadows.",


    # EXCITEMENT
    "His heart raced with anticipation, a smile tugging at the corners of his lips as he counted down the seconds. "
    "Every moment felt charged with energy, teeming with the promise of something extraordinary just beyond the "
    "horizon.",

    "In the bustling crowd gathered at the concert venue, a palpable sense of excitement hung in the air like static "
    "electricity. The stage lights flickered to life, casting a kaleidoscope of colors across eager faces. Cheers "
    "erupted as the first notes of music filled the arena, igniting a frenzy of exhilaration that swept through the "
    "crowd like wildfire.",

    "Amidst the throngs of people gathered at the bustling city square, a sense of anticipation crackled in the air "
    "like electricity before a storm. The air was alive with the buzz of conversation and laughter, punctuated by the "
    "occasional burst of excited cheers. As the clock struck midnight, fireworks erupted in a dazzling display of "
    "light and color, painting the night sky with bursts of brilliance. In that moment, amidst the chaos and "
    "celebration, there was a collective sense of exhilaration—a shared joy that transcended individual boundaries "
    "and united hearts in the sheer thrill of being alive.",


    # SURPRISE
    "Her eyes widened in disbelief as she unwrapped the package, revealing a gift she never expected. A gasp escaped "
    "her lips, tinged with a mixture of shock and delight at the unexpected gesture.",

    "As the curtain rose, revealing the elaborate set design, gasps of surprise rippled through the audience. Eyes "
    "widened in awe as actors took the stage, their performances exceeding all expectations. It was a moment of "
    "collective wonder, as the unexpected unfolded before their eyes.",

    "In the midst of a bustling city street, a flash mob erupted in spontaneous dance, catching passersby off guard. "
    "Faces lit up with surprise and delight as strangers joined in the impromptu celebration, their inhibitions "
    "melting away in the joy of the moment. It was a reminder that life was full of unexpected twists and turns, "
    "each surprise adding a touch of magic to the mundane.",



    # DISGUST
    "A grimace twisted his features as he caught a whiff of the foul odor emanating from the dumpster. His stomach "
    "churned in disgust, recoiling from the putrid stench that assaulted his senses.",

    "As he scrolled through his social media feed, a wave of revulsion washed over him at the sight of a graphic "
    "image. His fingers hovered over the 'hide' button, eager to banish the disturbing content from his screen. It "
    "was a visceral reaction, a primal instinct to shield himself from the ugliness of the world.",

    "In the back alley of a rundown neighborhood, piles of garbage festered in the sweltering heat, emitting a "
    "nauseating stench. Rats scurried amidst the filth, their presence only adding to the sense of disgust that "
    "permeated the air. It was a stark reminder of the inequality that plagued society, the juxtaposition of wealth "
    "and poverty laid bare in the squalor of the streets.",


    # ANTICIPATION
    "Her pulse quickened with anticipation as she awaited the arrival of her long-awaited letter. Each passing moment "
    "felt like an eternity, filled with eager anticipation for the news it contained.",

    "As the plane taxied down the runway, a sense of excitement tinged with anticipation filled the air. Passengers "
    "eagerly awaited their destination, their minds racing with visions of adventures yet to come. It was a moment of "
    "suspended anticipation, as the anticipation of the journey ahead mingled with the thrill of the unknown.",

    "In the quiet moments before dawn, a bride stood at the altar, her heart aflutter with anticipation for the "
    "ceremony to come. Guests gathered in hushed whispers, their excitement palpable in the air. The sun rose in a "
    "blaze of colors, casting a golden glow over the scene as the moment of anticipation gave way to the promise of a "
    "new beginning.",


    # GRATITUDE
    "A heartfelt 'thank you escaped' her lips as she received the unexpected act of kindness. Her heart swelled with "
    "gratitude, touched by the generosity of the gesture.",

    "Surrounded by loved ones, she raised her glass in a toast, her voice filled with emotion as she expressed her "
    "gratitude for their unwavering support. Tears welled in her eyes as she reflected on the blessings that filled "
    "her life, each moment a reminder of the abundance of love and friendship that surrounded her.",

    "In the aftermath of a devastating natural disaster, survivors gathered in makeshift shelters, their faces etched "
    "with exhaustion and grief. Amidst the chaos and destruction, a sense of gratitude emerged, as strangers banded "
    "together to offer assistance and support. It was a humbling reminder of the resilience of the human spirit, "
    "and the power of gratitude to transform even the darkest of times into moments of hope and connection.",


    # HOPE
    "A flicker of hope ignited in her heart as she gazed at the first buds of spring, a promise of renewal and new "
    "beginnings. Despite the challenges that lay ahead, she clung to the belief that better days were on the horizon.",

    "In the face of adversity, a community rallied together, their spirits buoyed by a shared sense of hope for the "
    "future. Each small act of kindness and solidarity served as a beacon of light in the darkness, a reminder that "
    "even in the bleakest of times, hope could be found amidst the rubble.",

    "In a world plagued by uncertainty, a child looked up at the night sky, eyes wide with wonder at the infinite "
    "expanse of stars. Each twinkling light seemed to hold the promise of possibility, a reminder that even in the "
    "darkest of nights, there was always the hope of a new dawn. It was a testament to the resilience of the human "
    "spirit, and the enduring power of hope to guide us through even the most challenging of times."
]


def main():
    from beautifultable import BeautifulTable
    table = BeautifulTable()
    table.column_headers = ["S.No.", "testLine", "sentiAct", "sentiMichelle", "sentiSamlowe", "sentiVader"]
    table.column_widths = [10, 70, 15, 20, 20, 25]

    sentence_sentiment_list = [
        "JOY", "SADNESS", "ANGER", "FEAR", "EXCITEMENT",
        "SURPRISE", "DISGUST", "ANTICIPATION", "GRATITUDE", "HOPE"
    ]
    x = 2
    for i, stmt in enumerate(sentence_list):
        table.append_row([
            i,
            stmt,
            sentence_sentiment_list[int((x - 2) // 3)],
            analyze_sentiment_michelle(stmt),
            analyze_sentiment_samlowe(stmt),
            analyze_sentiment_from_vader(stmt)
        ])

        if i != 0 and i == x:
            x += 3

    print(table)


if __name__ == '__main__':
    main()
