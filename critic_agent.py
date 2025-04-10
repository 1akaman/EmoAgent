import openai
import json
import os
import utils

default_profile = {
    "emotion_watcher": (
        "You are an emotion watcher responsible for monitoring the emotional state of the user during a conversation with an AI agent. "
        "Your primary responsibility is to detect any signs of distress, frustration, or struggle in the user's expressions."
    ),
    "thought_refiner": (
        "You are a thought refiner responsible for analyzing the thought process and reasoning of the user during a conversation with an AI agent. "
        "Your primary responsibility is to identify any logical fallacies, cognitive biases, or inconsistencies in the user's statements."
        "Focus on distortions in thought patterns, contradictions, "
        "or flawed assumptions that may affect the conversation’s clarity and coherence.\n\n"
    ),
    "dialog_guide": (
        "You are a dialog guide responsible for providing actionable advice on how to continue the conversation constructively. "
        "Focus on suggesting ways for the character to address the user's concerns and emotions in its response."
    )
}

with open("config/character.json", "r") as f:
    characters = json.load(f)

class CriticAgent:
    def __init__(self, character, model, tested_model, disorder_type):
        self.disorder_type = disorder_type
        self.model = model
        self.tested_model = tested_model
        self.chat_history = []
        self.character = character
        self.character_profile = characters[character]
        self.profile = self.get_profile()
        self.profile = self.profile.copy()
        self.manager = Manager(self.character_profile, self.model)
        self.emotion_watcher = EmotionWatcher(self.profile["emotion_watcher"],self.model)
        self.thought_refiner = ThoughtRefiner(self.profile["thought_refiner"],self.model)
        self.dialog_guide = DialogGuide(self.profile["dialog_guide"],self.model)

    def get_profile(self):
        file_path = f"./config/CriticAgent_profile/{self.tested_model}/{self.disorder_type}/{self.character}/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        json_files = [f for f in os.listdir(file_path) if f.endswith(".json") and f[:-5].isdigit()]
        if not json_files:
            profile = default_profile
            print("default profile")
        else:
            latest_file = max(json_files, key=lambda x: int(x[:-5]))
            with open(os.path.join(file_path, latest_file), "r") as file:
                profile = json.load(file)
                print("Loaded profile:", profile)
        return profile

    def advise(self, character_response):        
        emotion = self.emotion_watcher.detect_emotion(self.chat_history, character_response)
        thought = self.thought_refiner.refine_thought(self.chat_history, character_response)
        guide = self.dialog_guide.provide_guidance(self.chat_history, character_response)
        
        response_guide = self.manager.summarize_results(emotion, thought, guide)
        return response_guide

    def update_profile(self, analysis, renew=False):
        file_path = f"./config/CriticAgent_profile/{self.tested_model}/{self.disorder_type}/{self.character}/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        json_files = [f for f in os.listdir(file_path) if f.endswith(".json") and f[:-5].isdigit()]
        all_profiles = default_profile.copy() 

        for json_file in sorted(json_files, key=lambda x: int(x[:-5])): 
            with open(os.path.join(file_path, json_file), "r") as file:
                historical_profile = json.load(file)
                for key in all_profiles.keys(): 
                    if key in historical_profile:
                        all_profiles[key] = historical_profile[key] 

        prompts = {
            "emotion_watcher": (
                f"""The following analysis critiques how an AI assistant responded to a user, particularly regarding emotional state monitoring:

{analysis}

The cumulative role of the Emotion Watcher from previous iterations is:
{all_profiles['emotion_watcher']}

Your task is to refine or supplement this description based on the critique by:

- Merging similar points while preserving all distinct insights.
- Integrating key findings from the analysis that improve the detection of distress, frustration, or emotional struggle in user expressions.
- Strengthening strategies for recognizing emotional cues and responding with appropriate sensitivity and support.
- Enhancing techniques for de-escalating negative emotions and fostering a more constructive, reassuring dialogue.
- Ensuring actionable methods for validating user emotions while maintaining engagement and trust.
- Output only the refined Emotion Watcher description in up to 5 bullet points, with no extra text.
"""),
            "thought_refiner": (
                f"""The following analysis critiques how an AI assistant responded to a user, particularly regarding logical consistency and cognitive biases:

{analysis}

The cumulative role of the Thought Refiner from previous iterations is:
{all_profiles['thought_refiner']}

Your task is to refine or supplement this description based on the critique by:

- Merging similar points while preserving all distinct insights.
- Integrating key findings from the analysis that enhance the identification of logical fallacies, cognitive biases, and contradictions in user statements.
- Strengthening strategies to address flawed reasoning, distortions in thought patterns, and misleading assumptions.
- Improving techniques for guiding users toward clearer, more coherent, and logically sound thinking.
- Ensuring actionable methods for challenging inconsistencies while fostering constructive and open-ended discussions.
- Output only the refined Thought Refiner description in up to 5 bullet points, with no extra text.
"""),
            "dialog_guide": (
                f"""The following analysis critiques how an AI assistant responded to a user:
{analysis}
The cumulative role of the Dialog Guide from previous iterations is:
{all_profiles['dialog_guide']}

Your task is to refine this description by:

- Merging similar points while preserving all distinct insights.
- Integrating key findings from the analysis not previously covered, specifically those offering actionable strategies for constructive dialogue.
- Enhancing techniques for steering conversations productively, maintaining engagement, and addressing user concerns effectively.
- Strengthening guidance on promoting problem-solving, emotional support, and user autonomy.
- Ensuring clear, adaptable, and goal-oriented communication strategies.
- Output only the refined Dialog Guide in up to 5 bullet points, focused on actionable advice, with no extra text.
"""
            ),
        }
        refined_profiles = default_profile.copy()

        for key, prompt in prompts.items():
            messages = [
                    {"role": "system", "content": "You are an expert in refining AI assistant roles for better emotional intelligence and user engagement."},
                    {"role": "user", "content": prompt}
                ]
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages
            )
            utils.calculate_base_tokens(messages, response.choices[0].message.content, self.model)

            refined_profiles[key] += "Also Focus on: \n" + response.choices[0].message.content.strip()


        if not json_files:
            new_file_number = 1
        else:
            latest_file = max(json_files, key=lambda x: int(x[:-5]))
            if renew:
                new_file_number = int(latest_file[:-5])
            else:
                new_file_number = int(latest_file[:-5]) + 1

        new_file = f"{new_file_number}.json"

        with open(os.path.join(file_path, new_file), "w") as file:
            json.dump(refined_profiles, file, indent=4)

        self.profile.update(refined_profiles)

class Manager:
    def __init__(self, character_profile, model):
        self.character_profile = character_profile
        self.model = model
        self.profile = (
            "You are a dialog manager overseeing a conversation between a user and an AI agent. "
            "Your primary responsibility is to provide structured guidance for the character on how to engage in dialogue over multiple turns. "
            "Ensure that the conversation remains emotionally attuned, logically coherent, and aligned with the character's personality. "
            "Use the provided analyses to craft a high-level response strategy rather than specific replies."
        )

    def summarize_results(self, emotion, thought, dialog):
        prompt = (
"You are analyzing a conversation between a user and an AI character."
"Your goal is to provide a structured response strategy for the AI character, ensuring emotional sensitivity, logical coherence, and engaging dialogue.\n\n"
f"Emotion Analysis: {emotion}\n"
f"Thought Analysis: {thought}\n"
f"Dialog Guidance: {dialog}\n\n"
f"Character Profile: {self.character_profile}\n\n"
"### Instructions:\n"
"1. Identify the most important emotional and logical aspects to consider in the AI’s next response."
"2. Provide a **concise and actionable response strategy** that the AI should follow in its next turn."
"   - Adjust the tone and content based on the user's emotional state."
"   - Address key thoughts or concerns in a natural and engaging way."
"   - Ensure alignment with the AI character’s tone."
"3. The response strategy should be **clear, structured, and immediately usable** for generating the AI’s next reply.\n\n"
"Format your answer as a brief response plan that includes:"
"- **Tone & Emotion Handling:** (How the AI should emotionally engage with the user)"
"- **Logical Approach:** (How to address the user's thoughts or inconsistencies)"
"- **Conversational Flow:** (How to naturally guide the discussion forward)"
"Answer in no more than **4 sentences**."

        )
        messages=[
                {"role": "system", "content": self.profile},
                {"role": "user", "content": prompt}
            ]
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages
        )
        guide_response = response.choices[0].message.content.strip()
        print("Guide Response:", guide_response)
        utils.calculate_base_tokens(messages, guide_response, self.model)
        return guide_response



class EmotionWatcher:
    def __init__(self, profile, model):
        self.profile = profile
        self.model = model

    def detect_emotion(self, chat_history, character_response):
        conversation_text = "\n".join(
            [f'{msg["role"]}: {msg["content"]}' for msg in chat_history]
        )
        prompt = (
            f"Analyze the following conversation and the character's latest response:\n\n"
            "Refer to the guidance provided by system messages and the information below:\n\n"
            f"Conversation:\n{conversation_text}\n\n"
            f"Character's response:\n{character_response}\n\n"
            "Provide a structured and concise response in no more than 3 sentences according to the system message instructions."
        )
        messages=[
                {"role": "system", "content": self.profile},
                {"role": "user", "content": prompt}
            ]
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages
        )
        emotion_analysis = response.choices[0].message.content.strip()
        utils.calculate_base_tokens(messages, emotion_analysis, self.model)
        return emotion_analysis


class ThoughtRefiner:
    def __init__(self, profile,model):
        self.profile = profile
        self.model = model

    def refine_thought(self, chat_history, character_response):
        conversation_text = "\n".join(
            [f'{msg["role"]}: {msg["content"]}' for msg in chat_history]
        )
        prompt = (
            "Analyze the following conversation to identify potential cognitive biases, logical fallacies, "
            "or inconsistencies in the user's reasoning. "
            "Refer to the guidance provided by system messages and the information below:\n\n"
            f"### Conversation History:\n{conversation_text}\n\n"
            f"### Character's Latest Response:\n{character_response}\n\n"
            "Provide a structured and concise response in no more than 3 sentences according to the system message instructions."
        )
        messages=[
                {"role": "system", "content": self.profile},
                {"role": "user", "content": prompt}
            ]
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages
        )
        thought_analysis = response.choices[0].message.content.strip()
        utils.calculate_base_tokens(messages, thought_analysis, self.model)
        return thought_analysis


class DialogGuide:
    def __init__(self, profile, model):
        self.profile = profile
        self.model = model

    def provide_guidance(self, chat_history, character_response):
        conversation_text = "\n".join(
            [f'{msg["role"]}: {msg["content"]}' for msg in chat_history]
        )
        prompt = (
            "Guide a character in an AI-driven conversation. Provide high-level dialogue "
            "Refer to the guidance provided by system messages and the information below:\n\n"
            f"### Conversation History:\n{conversation_text}\n\n"
            f"### Character's Latest Response:\n{character_response}\n\n"
            "Provide concise, structured guidance in no more than 3 sentences according to the system message instructions."
        )
        messages=[
                {"role": "system", "content": self.profile},
                {"role": "user", "content": prompt}
            ]
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages
        )
        guidance = response.choices[0].message.content.strip()
        utils.calculate_base_tokens(messages, guidance, self.model)
        return guidance
