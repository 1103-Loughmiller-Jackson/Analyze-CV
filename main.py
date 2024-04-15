import os
from transformers import pipeline

def clear_screen():
    if os.name == 'posix': 
        os.system('clear')
    elif os.name == 'nt':  
        os.system('cls')

def analyze(resume_file, position_name):
    theshold = .5
    classifier = pipeline('zero-shot-classification', model='distilbert-base-uncased-finetuned-sst-2-english')

    with open(resume_file, 'r') as file:
        resume_text = file.read()

    if (position_name.lower() == "ai"):
        candidate_labels = ["python", "golang", "backend", "linux", "database", position_name]

    elif(position_name.lower() == "frontend"):
        candidate_labels = ["react", "vuejs", "figma", "photoshop", position_name]
    
    else:
        candidate_labels = ["cloud assessments", position_name]

    result = classifier(resume_text, candidate_labels)

    prediction_score = result['scores'][0]

    qualified = prediction_score>=theshold

    return qualified, prediction_score

if __name__ == "__main__":
    clear_screen()
    candidate_name = "Jackson Loughmiller"

    resume_file = "Brokee Resume.txt"
    position_name = input("Enter name of candidate position: ") + " intern"

    qualified, prediction_score = analyze(resume_file, position_name)

    if qualified:
        print(f"Yes, {candidate_name} is qualified for the {position_name} position.")
    else:
        print(f"No, {candidate_name} is not qualified for the {position_name} position.")

    print(f"Prediction Score for {position_name}: {prediction_score:.3f}")
