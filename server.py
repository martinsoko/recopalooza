from flask import Flask, request, jsonify
import recopalooza

app = Flask(__name__)


@app.route('/get_roadmap', methods=['GET'])
def generate_roadmap():
    artists = request.args.get('artists', type=str)
    artists = [artist.strip() for artist in artists.split(',')]
    audio_feats = request.args.get('audio_feats', type=str)
    audio_feats = [feat.strip() for feat in audio_feats.split(',')]
    shuffle = request.args.get('shuffle', type=int)
    roadmap = recommender.get_roadmap(artists, audio_feats, shuffle)
    return jsonify({'roadmap': roadmap})


if __name__ == '__main__':
    recommender = recopalooza.Recommender(clusters_file='artists_lolla.csv')
    app.run()


