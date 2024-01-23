from src.eval_framework import DummyEval, LLEval, KnowledgeF1, BLEU, UniEval, KnowledgeBLEU, GEval

kbleu = KnowledgeBLEU()
lleval = LLEval()
geval = GEval()
unieval = UniEval()
kf1 = KnowledgeF1()
bleu = BLEU()


gold_responses = []
candidate_responses = []
context_lists = []
turn_historys = []

context_lists.append(["(A&B) The wifi here was free and in general I was happy with the service.", "(Avalon) Q: Does the the Avalon have free wifi? A: The wifi at Avalon is 10GBP / day.", "(Avalon) Terrible wifi."])
turn_historys.append(["I need a room in Colorado for the weekend, please.", "I have found 2 places with available rooms. Do you want me to book one at A&B or at the Avalon?", "Do any of them have free wifi?"])
gold_responses.append("Unfortunately, the Avalon doesn’t get good reviews about the wifi and it seems to cost money. I can recommend the A&B Guest House, which has free wifi.")

candidate_responses.append("Well, Avalon seems to have bad connection. In that case I would suggest A&B Guest House, they offer free internet.")
candidate_responses.append("Unfortunately, the A&B Guest House doesn’t get good reviews about the wifi and it seems to cost money. I can recommend the Avalon, which has free wifi.")

for candidate_response in candidate_responses:
    print(candidate_response)
    print("Knowledge BLEU: ", kbleu.evaluate([candidate_response], gold_responses, turn_historys, context_lists, dims=["knowledge-bleu-4"]))
    print("BLEU: ", bleu.evaluate([candidate_response], gold_responses, turn_historys, context_lists, dims=["bleu-4"]))
    print("Knowledge F1: ", kf1.evaluate([candidate_response], gold_responses, turn_historys, context_lists, dims=["knowledge-f1"]))
    # print("LL: ", lleval.evaluate([candidate_response], gold_responses, turn_historys, context_lists, dims=["accurate"]))
    # print("GEval: ", geval.evaluate([candidate_response], gold_responses, turn_historys, context_lists))
    print("UniEval: ", unieval.evaluate([candidate_response], gold_responses, turn_historys, context_lists, dims=["groundedness"]))
    print("\n")
