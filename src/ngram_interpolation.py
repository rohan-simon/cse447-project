from collections import defaultdict


class InterpolatedNGramModel:
    def __init__(self, text, max_order=3, lambdas=None, smoothing=1e-6):
        """
        Character-level n-gram language model with linear interpolation.

        Args:
            text: Training corpus as a single string.
            max_order: Highest n-gram order to use (e.g., 3 for trigram).
            lambdas: Interpolation weights from unigram..max_order.
                     Length must equal max_order and sum to 1.
                     If None, uses a default decay that favors higher orders.
            smoothing: Add-k smoothing constant for every conditional estimate.
        """
        self.text = text
        self.max_order = max_order
        self.smoothing = float(smoothing)

        if lambdas is None:
            base = [2 ** i for i in range(max_order)]
            total = sum(base)
            self.lambdas = [value / total for value in base]
        else:
            total = sum(lambdas)
            self.lambdas = [value / total for value in lambdas]

        self.vocab = []
        self.vocab_set = set()
        self.ngram_counts = {order: defaultdict(int) for order in range(1, max_order + 1)}
        self.context_counts = {order: defaultdict(int) for order in range(2, max_order + 1)}

    def fit(self):
        self.vocab_set = set(self.text)
        self.vocab = sorted(self.vocab_set)

        text_len = len(self.text)

        for order in range(1, self.max_order + 1):
            if text_len < order:
                continue

            for idx in range(order - 1, text_len):
                ngram = self.text[idx - order + 1 : idx + 1]
                self.ngram_counts[order][ngram] += 1

                if order > 1:
                    context = ngram[:-1]
                    self.context_counts[order][context] += 1

    def char_prob(self, char, context, order):
        """
        P(char | context) for a specific order using add-k smoothing.
        """
        vocab_size = len(self.vocab)
        k = self.smoothing

        if order == 1:
            numerator = self.ngram_counts[1].get(char, 0) + k
            denominator = len(self.text) + k * vocab_size
            return numerator / denominator

        short_context = context[-(order - 1) :] if order > 1 else ""
        ngram = short_context + char

        numerator = self.ngram_counts[order].get(ngram, 0) + k
        denominator = self.context_counts[order].get(short_context, 0) + k * vocab_size
        return numerator / denominator

    def score_next_char(self, prefix):
        scores = {}
        for char in self.vocab:
            probability = 0.0
            for order in range(1, self.max_order + 1):
                probability += self.lambdas[order - 1] * self.char_prob(char, prefix, order)
            scores[char] = probability
        return scores

    def predict(self, prefixes):
        """
        Returns top-3 character guesses for each prefix.
        """
        predictions = []
        for prefix in prefixes:
            scores = self.score_next_char(prefix)
            top_chars = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
            predictions.append("".join(char for char, _ in top_chars))
        return predictions
