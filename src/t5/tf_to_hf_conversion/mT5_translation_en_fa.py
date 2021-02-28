from transformers import T5Config, MT5ForConditionalGeneration, MT5Tokenizer
from transformers.models.t5.modeling_t5 import load_tf_weights_in_t5

if True:
    size = "large"
    model_name = f"google/mt5-{size}"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration(T5Config.from_pretrained(model_name))

    load_tf_weights_in_t5(model, None, f"/Users/danielk/ideaProjects/parsiglue-baselines/src/t5/tf_to_hf_conversion/{size}")
    model.eval()

    model.save_pretrained(f"/Users/danielk/ideaProjects/parsiglue-baselines/src/t5/tf_to_hf_conversion/mt5-{size}-parsinlu-translation_en_fa")
    tokenizer.save_pretrained(f"/Users/danielk/ideaProjects/parsiglue-baselines/src/t5/tf_to_hf_conversion/mt5-{size}-parsinlu-translation_en_fa")
else:
    model_name = "/Users/danielk/ideaProjects/parsiglue-baselines/huggingface_example_scripts/mT5_persian_multiple_choice_small"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    print(output)
    return output


run_model("Praise be to Allah, the Cherisher and Sustainer of the worlds;")
run_model("shrouds herself in white and walks penitentially disguised as brotherly love through factories and parliaments; offers help, but desires power;")
run_model("He thanked all fellow bloggers and organizations that showed support.")
run_model("Races are held between April and December at the Veliefendi Hippodrome near Bakerky, 15 km (9 miles) west of Istanbul.")
run_model("I want to pursue PhD in Computer Science about social network,what is the open problem in social networks?")

