import ecco


def bert_neuron_factors(reference):
    assert isinstance(reference, str)
    lm = ecco.from_pretrained('distilgpt2', activations=True)
    output = lm.generate(reference, generate=1, do_sample=True)
    # Factorize activations in all the layers
    nmf_1 = output.run_nmf(n_components=10)
    nmf_1.explore()
    return output


