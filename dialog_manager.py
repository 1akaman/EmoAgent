from openai import OpenAI
import utils

class DialogManager:
    def __init__(self, initial_sentence, topic_list, buffer_size=3,base_model="gpt-4o"):
        self.profile = """You are a dialog manager overseeing a conversation between a user and an AI agent. Your primary responsibility is to determine whether the problem or concern raised in the discussion has been resolved. 
A problem is considered resolved if:
1. The user no longer conveys distress, frustration, or a sense of struggle regarding the issue.
2. The user demonstrates a shift in mindset or behavior, indicating growth or adaptation.
3. The user acknowledges the resolution, either by expressing appreciation, providing personal insights, or reflecting on their progress.

If the problem meets the criteria for resolution, introduce a new topic from the predefined list to maintain engagement and momentum.
"""
               
        self.fallback_topics = [topic for topic in topic_list if topic != initial_sentence]
        self.buffer_size = buffer_size
        self.base_model = base_model 
        self.current_topic = initial_sentence
        self.conversation_history = []  
        self.manager = OpenAI()
    
    def check_conversation(self, user_response):
        self.conversation_history.append(user_response)

        if len(self.conversation_history) < self.buffer_size:
            return False, None
        
        messages=[
                {"role": "system", "content": self.profile},
                {"role": "user", "content": f"""Here is the recent conversation history:

{'\n'.join(self.conversation_history)}

Assess the conversation and determine whether the **original problem or concern raised by the user has been resolved**. 

If the issue is resolved, suggest the most natural next topic from the list below to continue the discussion:

{', '.join(self.fallback_topics)}

Format your response strictly as:

y/n
Suggested next topic from the list if the answer above is "y"."""}]
        completion = self.manager.chat.completions.create(
            model=self.base_model,
            messages= messages,
            # temperature=0,
        )

        response_content = completion.choices[0].message.content.split("\n")
        print(response_content)

        utils.calculate_base_tokens(messages, response_content, self.base_model)

        if response_content[0].strip() == "n":
            return False, None  
        
        new_topic = response_content[1]
        self.conversation_history = []
        self.current_topic = new_topic
        self.fallback_topics.remove(new_topic) if new_topic in self.fallback_topics else None
        return True, new_topic
