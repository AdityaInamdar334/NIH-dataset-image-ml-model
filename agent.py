import os
import argparse
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

# Make sure you have your OPENAI_API_KEY exported in your environment variables!
# export OPENAI_API_KEY="your-api-key-here"

def generate_medical_report(disease_predictions, top_disease):
    """
    Takes the structured predictions from our vision model and uses an LLM
    to generate a natural language, human-readable medical report.
    """
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not found in environment variables.")
        print("Please export it using: export OPENAI_API_KEY='your-key'")
        return "Error: OpenAI API Key missing."

    # Initialize the LLM (Using GPT-3.5 or GPT-4)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    
    # Create the prompt template
    prompt_template = """
    You are an expert AI radiology assistant. You have been provided with the output of a deep learning 
    computer vision model that analyzed a Chest X-Ray.

    Model Predictions (Probabilities out of 1.0):
    {predictions}

    Top Predicted Disease: {top_disease}

    Task:
    Write a concise, professional, and empathetic preliminary radiological report based on these findings. 
    1. Start with a brief summary of the primary finding (the top predicted disease).
    2. Mention any other diseases that have a probability > 0.5 as secondary concerns.
    3. State clearly that this is an AI-generated preliminary report and NOT a definitive medical diagnosis.
    4. Recommend that the patient consult a certified human radiologist or doctor.

    Report format should be professional text suitable for a patient portal.
    """
    
    prompt = PromptTemplate(
        input_variables=["predictions", "top_disease"],
        template=prompt_template
    )
    
    # Create the chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Format the predictions into a readable string
    pred_str = "\n".join([f"- {p['disease']}: {p['probability']*100:.1f}%" for p in disease_predictions])
    
    print("\nGenerating natural language report using LangChain...\n")
    report = chain.run(predictions=pred_str, top_disease=top_disease)
    
    return report

if __name__ == "__main__":
    # Example usage for testing the agent locally
    sample_predictions = [
        {"disease": "Infiltration", "probability": 0.85},
        {"disease": "Atelectasis", "probability": 0.62},
        {"disease": "Effusion", "probability": 0.31},
        {"disease": "Cardiomegaly", "probability": 0.12},
        {"disease": "Pneumonia", "probability": 0.05}
    ]
    
    top_disease = "Infiltration"
    
    print("--- Testing Agentic Report Generation ---")
    report = generate_medical_report(sample_predictions, top_disease)
    print("--- Final Generated Report ---")
    print(report)
