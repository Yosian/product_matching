import pandas as pd
from matching_items_code.product_matching.entities import import_data


class Explore:
    def __init__(self):
        self.frame = import_data()

    def nan_values(self) -> None:
        """ How much of the data are missing values """
        print('Fraction of the data that is missing from each column\n')
        print(self.frame.isnull().sum() / len(self.frame))
        print('\n Most columns can be considered reliable. We may disregard the column '
              'brand_name')

    def _add_word_count(self) -> None:
        """ Adds a dummy column in the dataframe to count the number of words in the
        name"""
        if 'num_words' not in self.frame.columns:
            self.frame['num_words'] = [len(field.split(' ')) for field in self.frame.name]

    def _uniqueness_by_word_number(self, words: int) -> None:
        self._add_word_count()
        temp = self.frame.loc[self.frame.num_words == words].copy()
        print(f'The following breakdown shows the 10 most common repeated field'
              f'name containing {words} words\n')
        print(temp.name.value_counts()[:10])

    def uniqueness(self, number_cases: int = 10) -> None:
        """ This function checks if users are consistent in typing data. It also describes
        what groups of items are described consistently. The uniqueness is focused in name
        as I consider it the most relevant field to study repetitions """
        print(f'The following breakdown shows the {number_cases} most common repeated '
              f'field name\n')
        print(self.frame.name.value_counts()[:number_cases])
        print('\nMost field name listed are monosyllables and not specific enough to '
              'match these items')

        self._uniqueness_by_word_number(words=3)
        self._uniqueness_by_word_number(words=5)
        self._uniqueness_by_word_number(words=8)
        print('\nWith higher word count the name field becomes more significant. I will '
              'probably make the number of words a variable, so that the higher number of'
              ' matching words translates into a more reliable prediction')

    # def price_correlation(self):
    #     """ This function tries to give an insight into the level of matching among items
    #     with the same name. I expect longer matching names are very similar if not the
    #     same. Since there is no label to tell whether two items match, the price will be
    #     used as a validator. """
    #
    #     self._add_word_count()
    #     # first select only the name, num_words and price columns
    #     temp = self.frame[['name', 'num_words', 'price']].copy()
    #     # count how many times the name repeats
    #     temp['repetitions'] = temp.groupby(['name'])['price'].transform(lambda x: len(x))
    #     # use names with over 100 repetitions so the correlation is significant
    #     temp = temp.loc[temp.repetitions > 50]
    #     # observe if there is any correlation and if it changes with a greater matching
    #     # number of words
    #     for i in range(max(temp.num_words)):
    #         print(f'Correlation between price and number of words when num_words = {i}\n')
    #         _ = temp.loc[temp.num_words == i]
    #         print(_.num_words.corr(_.price))
    #
    #     print('No correlation observed between price and number of matching words, this means'
    #           ' the price is not a validator to know if two items match, or the number of '
    #           'matching words does not tell us if two items are the same, or both statements '
    #           'are true')
