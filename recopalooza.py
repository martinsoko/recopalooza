import pandas as pd
import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import euclidean
import argparse


class Recommender:
    # General attributes
    artist_audio_feats_ = pd.DataFrame()
    grid_ = pd.DataFrame()
    clusters_ = pd.DataFrame()
    verbosity_ = False
    # User specific attributes
    chosen_artists = []
    chosen_clusters = pd.DataFrame()
    feature_weights = []
    shuffleness = 0

    def __init__(self, grid_file='horarios.csv',
                 clusters_file='clusters.csv',
                 audio_feats_file='artists_audio_feats.csv',
                 verbosity=False):
        # Load necessary files
        self.artist_audio_feats_ = pd.read_csv(audio_feats_file, index_col='artist')
        self.clusters_ = pd.read_csv(clusters_file, index_col='artist')
        self.grid_ = pd.read_csv(grid_file)
        self.verbosity_ = verbosity

    def fill_slot(self, slot):
        options = self.grid_.iloc[slot, 1: -1].dropna()
        if self.verbosity_:
            print('Choosing between ', options.tolist())
        # if an option is in the chosen bands
        if options.isin(self.chosen_artists).sum() == 1:
            # fill with chosen artist
            chosen = options[options.isin(self.chosen_artists)].values[0]
            if self.verbosity_:
                print(chosen, ' is among the chosen artists.')
            return chosen
        else:
            if self.verbosity_:
                print('There are no chosen artists among the options. Choosing...')
            chosen = self.choose_among_options(options)
            if self.verbosity_:
                print('Chose ', chosen)
            return chosen

    def get_roadmap(self, chosen_artists, feature_order, shuffleness):
        self.chosen_artists = chosen_artists
        self.feature_weights = feature_order
        self.chosen_clusters = self.clusters_.loc[chosen_artists, 'cluster_name']. \
            value_counts().rename('cluster_weight')
        roadmap = []
        if len(chosen_artists) == 0:
            self.shuffleness = 2
        else:
            self.shuffleness = shuffleness

        for i in range(len(self.grid_)):
            if self.verbosity_:
                print('Filling slot ', i)
            roadmap.append(self.fill_slot(i))
        return roadmap

    def choose_among_options(self, options):
        if self.shuffleness != 2:
            if self.verbosity_:
                print('\tVoting based on chosen clusters...')
            # if there are options in the same clusters as the chosen artists' clusters,
            # all the chosen artists vote for their cluster
            votes = pd.merge(self.clusters_.loc[options, 'cluster_name'],
                             self.chosen_clusters,
                             left_on='cluster_name',
                             right_index=True,
                             ).sort_values('cluster_weight', ascending=False)
            # if there are votes
            if len(votes) > 0:
                # if there's only artist voted
                if len(votes) == 1:
                    # return the winner
                    if self.verbosity_:
                        print('\tThere is a winner.')
                    return votes.index[0]
                # if there is an artist with more votes than others
                elif votes['cluster_weight'].iloc[0] != votes['cluster_weight'].iloc[1]:
                    # return the winner
                    if self.verbosity_:
                        print('\tThere is a winner.')
                    return votes.index[0]
                else:
                    # if there's a tie between the winners
                    winners = votes.index[votes['cluster_weight'] == votes['cluster_weight'].iloc[0]].tolist()
                    # Measure distances
                    artists, distances = self.distances_to_features(winners)
                    if self.shuffleness == 0:
                        winner = artists[np.argmin(distances)]
                    else:
                        winner = np.random.choice(artists, p=softmax(1-np.array(distances)))
                    if self.verbosity_:
                        print('\t', winner, ' won')
                    return winner
            # if there are no votes
            if self.verbosity_:
                print('\tThere are no votes. Solving ties between all options...')
            artists, distances = self.distances_to_features(options)
            if self.shuffleness == 0:
                winner = artists[np.argmin(distances)]
            else:
                winner = np.random.choice(artists, p=softmax(1 - np.array(distances)))
            if self.verbosity_:
                print('\t', winner, ' won')
            return winner
        else:
            if self.verbosity_:
                print('\tChoosing randomly between options')
            winner = np.random.choice(options)
            if self.verbosity_:
                print('\t', winner, ' chosen')

    def distances_to_features(self, options):
        w = [1.0, 0.5, 0.25, 0.125]
        distances = []
        artists = []
        for option in options:
            u = self.artist_audio_feats_.loc[option, self.feature_weights]
            for chosen_artist in self.chosen_artists:
                v = self.artist_audio_feats_.loc[chosen_artist, self.feature_weights]
                distances.append(euclidean(u, v, w))
                artists.append(option)
        return artists, distances


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--grid_file', required=False, help="Path to csv with the time grid")
    parser.add_argument('-c', '--clusters_file', required=False, help="Path to csv with clusters")
    parser.add_argument('-a', '--audio_feats_file', required=False, help='Path to csv with audio features')
    parser.add_argument('-r', '--artists', required=True, help='List of chosen artists', type=str)
    parser.add_argument('-f', '--audio_feats', required=True, help='Audio features sorted by preference', type=str)
    parser.add_argument('-s', '--shuffle', required=True, help='Shuffleness level', type=int)
    args = parser.parse_args()

    artists = [artist.strip() for artist in args.artists.split(',')]
    audio_feats = [feat.strip() for feat in args.audio_feats.split(',')]

    reco = Recommender()
    roadmap = reco.get_roadmap(artists, audio_feats, args.shuffle)
    print(roadmap)
