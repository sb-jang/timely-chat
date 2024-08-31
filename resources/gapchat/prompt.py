import json
import asyncio
import openai
from typing import Text
import re
import argparse

openai.api_key =""
model_name = "gpt-4o-mini"

def get_chatgpt_response(content: Text, task: Text):
    prompt = ""
    if task == "events":
        #if speaker_1 has... 는 내가 추가한 부분
        prompt = """In the following conversation, speaker_1 and speaker_2 are updating their daily lives. 
        In the conversation, both speakers might have mentioned some events. 
        The events might have been finished or are currently going on and just started. 
        Extract only the events that the speakers are currently going on and just started. 
        Summarize the events as nouns or noun phrases, such as "going for a tirp", "starting a MBA program", "taking an online course", "building a swimming pool".
        Describe the events as brief as possible using the shortest summary.
        Generate the answers in the format of "speaker_1: <event_1>, <event_2>.\nspeaker_2: <event_1>, <event_2>".
        if speaker_1 has the events for "going for a trip" and speaker_2 has the events for "starting a MBA program", generate "speaker_1: going for a trip.\nspeaker_2: starting a MBA program."
        "speaker_1" and "speaker_2" must be in lowercase, and there must be a "\n" before "speaker_2".
        If the speaker did not mention any events, generate: "<speaker>: Not mentioned."\nConversation: \n"""+ content 
    elif task == "schedule":
        prompt = """
        Given a list of events, generate a short schedule for finishing each event in JSON format.
        If it requires more information to get the schedule, roughly estimate one.
        Generate the answer with the following requirements:
        1. Must be a valid json file that can be parsed by python json package. Pay attention to the commas. 
        2. The format should be the same as the Answer shown in the #Example.
        3. Each field in the Answer is a list.
        #Example:
        events: 
        {
            "speaker_1": [
                "just started a one-year collage program."
            ],
            "speaker_2": [
                "just started taking an online course in data science",
                "getting driving license",
                "planning to move to a new house or apartment in the next 6 months."
            ]
        }
        Answer: 
        {
            "speaker_1": [
                "1 month for initiating, 2 months for basic courses, 3 months for main courses, 2 months for selecting thesis topics, 2 months for finishing thesis, 1 month for preparing defense."
            ],
            "speaker_2": [
                "1 week for learning basics, 2 weeks for learning programming techniques, 3 weeks for finishing tutorial projects, 2 weeks for finishing a final test",
                "1 week for learning rules, 2 weeks for practicing, 2 weeks for passing exams, 1 week for road check, 1 week for getting license",
                "2 weeks for searching apartment, 1 week for checking house and signing contract, 2 weeks for moving, 2 weeks for cleaning, 2 weeks for buying utilities."
            ]
        }

        # Question:
        List of events:
        """ + content
    elif task == "duration":
        prompt = """
        Task: estimate time duration. 
        Given a list of events provided by two people, speaker_1 and speaker_2, estimate a typical time duration to finish each event.
        For each event, label it with a time duration tag in the following steps:
        1. select a base time range from {hour, day, week, month, year} using the commonsense knowledge.
        2. select a number that is associated with the base time range to form the final time duration tag.
        3. generate the answer with <speaker_id>: <event> -> <number><base time range>.
        4. use N/A if the events for a speaker are not provided.
        Estimate the time duration for each event in the event list.
        The generated text contains only the answer.
        # Example:
        List of events:
        {
                "speaker_1": [
                    " just started a one-year collage program."
                ],
                "speaker_2": [
                    " just started taking an online course in data science",
                    " getting driving license",
                    " planning to move to a new house or apartment in the next 6 months."
                ]
        }
        Answer:
        speaker_1: just started a one-year collage program. -> 1 year
        speaker_2: just started taking an online course in data science -> 6 months, getting driving license -> 2 months, planning to move to a new house or apartment in the next 6 months. -> 6 months

        List of events:
        {
                "speaker_2": [
                    " Plan to have vacation for 5 days",
                    " plan to go on a family trip for relaxation",
                    " planning for career success."
                ]
        }
        Answer:
        speaker_1: N/A
        speaker_2: Plan to have vacation for 5 days -> 5 days, plan to go on a family trip for relaxation -> 1 week, planning for career success. -> indefinite

        Question:
        List of events:
        """ + content

    else:
        print("wrong task name, select from _get_argsduration]")
        return
    log_file_path = f"./data/new_data/{args.split}/{task}_log.jsonl"
    log_file = open(log_file_path, "a+", encoding='utf-8')

    messages = {
        "role": "user", 
        "content": prompt}

    response = {}
    raw_text = ""
    if model_name == "gpt-3.5-turbo":
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[messages],

        )
        raw_text = response["choices"][0]["message"]["content"]
    elif model_name == "gpt-4o-mini":
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[messages],
        )
        raw_text = response["choices"][0]["message"]["content"]
    elif model_name == "text-davinci-003":
        response = openai.Completion.create(
            model=model_name,
            prompt=prompt,
            top_p=1
        )
        raw_text = response["choices"][0]["text"]
    log_file.write(json.dumps(response) + "\n")

    if task == "schedule":
        if model_name == "gpt-3.5-turbo":
            schedule = raw_text.replace('\n', '')
            print(f"Schedule: {schedule}")
            raw_text = raw_text.replace("Answer:\n", "")
            raw_text = raw_text.replace("Answer: \n", "")
        elif model_name == "gpt-4o-mini":
            matches = re.findall(r'\{[^{}]*\}', raw_text)
            raw_text = matches[0]
        # if fail to load JSON format, return "Fail"
        try:
            return json.loads(raw_text)
        except:
            return "Fail"
    
    if "Answer:" in raw_text:
        index = raw_text.find('speaker_1')
        if index != -1:
            raw_text = raw_text[index:]

    if raw_text == "":
        raw_text = "speaker_1: Not mentioned.\nspeaker_2: Not mentioned."
    if "\n" in raw_text: # it means that data has speaker_1 and speaker_2
        extracted_events = raw_text.split("\n")
    else:
        if ("speaker_1" in raw_text) and ("speaker_2" in raw_text):
            parts = raw_text.split("speaker_")
            extracted_events = [f"speaker_{part.strip()}" for part in parts if part]
        else:
            extracted_events = raw_text
        print(extracted_events)
    return extracted_events

def read_conversations(data_path: Text):
    file = open(data_path, 'r', encoding='utf-8')
    data = json.load(file)
    conversations = []
    for session in data:
        conversation = ""
        for each_session in session['sessions']:
            for utterance in each_session:
                conversation += utterance["speaker"] + ": " + utterance["text"] + "\n"
            # conversation is equal to each session
            conversations.append(conversation)
    print(len(conversations))
    return conversations

def read_log_data(data_path: Text):
    data = []
    with open(data_path, 'r') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:  # parsing if not white space or empty line
                try:
                    parsed_data = json.loads(stripped_line)
                    data.append(parsed_data)
                except json.JSONDecodeError as e:
                    print(f'Error parsing JSON: {e}')
    return data

def read_events(data_path: Text):
    file = open(data_path, 'r', encoding='utf-8')
    data = json.load(file)
    events = []
    for item in data:
        tmp_events = {
            "speaker_1": "",
            "speaker_2": ""
        }
        empty_1 = False
        empty_2 = False
        if "speaker_1" in item and item["speaker_1"] != ["Not mentioned."]:
            tmp_events["speaker_1"] = item["speaker_1"]
        else:
            empty_1 = True
        if "speaker_2" in item and item["speaker_2"] != ["Not mentioned."]:
            tmp_events["speaker_2"] = item["speaker_2"]
        else:
            empty_2 = True
        if empty_1 and empty_2:
            events.append("")
        else:
            events.append(json.dumps(tmp_events, indent=4))
    return events

def extract_events(data_path: Text):
    events_file = open(f"./new_data/{args.split}/extracted_events.json", 'a+', encoding='utf-8')
    events_list = []
    conversations = read_conversations(data_path)
    for conversation in conversations:
        print(conversation)
        extracted_events = get_chatgpt_response(conversation, task="events")
        print(extracted_events)
        speaker_1_event = "Not mentioned."
        speaker_2_event = "Not mentioned."
        for event in extracted_events:
            if "speaker_1" in event:
                speaker_1_event = event.split("speaker_1:")[1].strip()
            elif "speaker_2" in event:
                speaker_2_event = event.split("speaker_2:")[1].strip()
        events_list.append({
            "speaker_1" : speaker_1_event.split(", "),
            "speaker_2" : speaker_2_event.split(", "),
        })
    json.dump(events_list, events_file, indent=4)
    

def get_events_from_logs(log_path: Text):
    events_file = open(f"./new_data/{args.split}/extracted_events", 'a+', encoding='utf-8')
    events_list = []
    log_data = open(log_path, 'r', encoding='utf-8')
    counter = 0
    for line in log_data.readlines():
        print(counter)
        line_data = json.loads(line.strip())
        response = line_data["choices"][0]["message"]["content"]
        extracted_events = response.split("\n")
        print(extracted_events)
        speaker_1_event = " Not mentioned."
        speaker_2_event = " Not mentioned."
        for event in extracted_events:
            if "speaker_1" in event:
                speaker_1_event = event.split("speaker_1:")[1].strip()
            elif "speaker_2" in event:
                speaker_2_event = event.split("speaker_2:")[1].strip()
        speaker_1_event_lst = [i.strip() for i in speaker_1_event.split(",")]
        speaker_2_event_lst = [i.strip() for i in speaker_2_event.split(",")]
        events_list.append({
            "speaker_1" : speaker_1_event_lst,
            "speaker_2" : speaker_2_event_lst,
        })
        counter += 1
    json.dump(events_list, events_file, indent=4)


def estimate_time(event_path: Text):
    time_tagged_file = open(f"./new_data/{args.split}/time_tag.jsonl", 'a+', encoding='utf-8')
    tagged_list = []
    counter = 500
    events = read_events(event_path)
    for event in events[500:]:
        time_tag = {
            "speaker_1": [],
            "speaker_2": []
        }
        if event == "":
            print(f"Skipping {counter}")
        elif event != "":
            print(f"Processing {counter}")
            extracted_time = get_chatgpt_response(event, task="duration")
            if len(extracted_time) == 2:
                for time in extracted_time:
                    for part in time.split(":")[1].split(","): # one event in a specific speaker
                        if "->" in part:
                            time_tag[time.split(":")[0].replace(' ', '').lower()].append(part.split("->")[1].strip())
            elif len(extracted_time) == 1:
                for part in time.split(":")[1].split(","):
                    if "->" in part:
                        time_tag[time.split(":")[0].replace(' ', '').lower()].append(part.split("->")[1].strip())
        tagged_list.append(time_tag)
        counter += 1
        time_tagged_file.write(json.dumps(time_tag) + "\n")


def get_schedule(event_path: Text):
    event_schedule_file = open(f"./new_data/{args.split}/schedule.jsonl", 'a+', encoding='utf-8')
    events = read_events(event_path)
    counter = 0
    print(len(events))
    for event in events:
        print(json.dumps(event, indent=4))
        schedule = {
            "speaker_1": [],
            "speaker_2": []
        }
        if event == "":
            print(f"Skipping {counter}")
        elif event != "":
            print(f"Processing {counter}")
            event_schedule = get_chatgpt_response(event, task="schedule")
            if event_schedule == "Fail":
                print(counter,"fail")
            schedule = event_schedule
        counter += 1
        event_schedule_file.write(json.dumps(schedule) + "\n")


def get_evaluation_conversation(save_path):
    conversation_file = open(save_path, 'a+', encoding='utf-8')
    prompt = """
    You are having a multi-session conversation with another speaker with the following conditions and example.
    
    # conditions:
    1. You are updating your current daily events.
    2. You are aware of the rough time estimation to finish different events.
    3. The conversation contains 3 sessions. 
    4. There is a time gap between each session.

    # your events:
    {events}

    # time gap:
    {gap}
    """

    gap_prompt = """
    It's been {gap}, and you talk with 
    """

    log_file_path = f"./data/new_data/self_chatgpt_log.jsonl"
    log_file = open(log_file_path, "a+", encoding='utf-8')
    response = {}
    messages = {
        "role": "user", 
        "content": prompt}
    response = openai.ChatCompletion.create(
            model=model_name,
            messages=[messages]
    )
    log_file.write(json.dumps(response) + "\n")

    raw_text = response["choices"][0]["message"]["content"]
    response = {
        "dialog": raw_text
    }
    conversation_file.write(json.dumps(response) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default="train", choices=['train','eval'])
    args = parser.parse_args()
    
    data_path = f"./gap_chat/{args.split}/merge.json" # path to gap_chat data
    events_path = f"./new_data/{args.split}/extracted_events.json" # path to events

    read_conversations(data_path)
    extract_events(data_path)
    #log_path = f"./new_data/{args.split}/events_log.jsonl"
    #get_events_from_logs(log_path)
    estimate_time(events_path)
    get_schedule(events_path)

    # self_chat_gpt_path = r"./new_data/self_chat_gpt.jsonl"
    # for index in range(15):
    #     get_evaluation_conversation(self_chat_gpt_path)