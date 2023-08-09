from transformers import pipeline

model_checkpoint2 = r"D:\WORK DRIVE\CODING MATERIALS\Self Projects\NN and LLM course\Project MedBot\QuestBot_Project\model_checkpoint2"
question_answerer = pipeline("question-answering", model=model_checkpoint2)

question = "Who are the main superheroes in 'The Avengers'?"
context = "In the movie 'The Avengers', a team of superheroes, including Iron Man, Captain America, Thor, and the Hulk, comes together to save the world from a powerful villain named Loki."

question_answerer(question=question, context=context)

import gradio as gr
iface = gr.Interface(
    fn=lambda context, question: question_answerer(question=question, context=context)['answer'],
    inputs=[
        gr.inputs.Textbox(label="Context"),
        gr.inputs.Textbox(label="Question")
    ],
    outputs=gr.outputs.Textbox(label="Answer")
)

# Launch the interface
iface.launch(share=True)









