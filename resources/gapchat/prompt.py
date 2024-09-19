import json
import asyncio
import openai
from typing import Text, List, Union, Dict
import re
import argparse
from datasets import load_dataset
from huggingface_hub import login

openai.api_key =""
model_name = "gpt-4o-mini"

def get_chatgpt_response(content: str, task: str) -> Union[str, List[str]]:
    prompt = ""
    if task == "events":
        #if speaker_1 has... 는 내가 추가한 부분
        prompt = """In the following conversation, speaker_1 and speaker_2 are updating their daily lives. 
        In the conversation, both speakers might have mentioned some events. 
        The events might have been finished or are currently going on and just started. 
        Extract only the events that the speakers are currently going on and just started. 
        Summarize the events as nouns or noun phrases, such as "going for a trip", "starting a MBA program", "taking an online course", "building a swimming pool".
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
    if args.dataset == "gapchat":
        log_file_path = f"./new_data/{args.split}/{task}_log.jsonl"
    elif args.dataset == "timelychat":
        log_file_path = f"/home/minjinj/timely-chat/resources/data/new_data/{task}_log.jsonl"
    log_file = open(log_file_path, "a+", encoding='utf-8')

    messages = {
        "role": "user", 
        "content": prompt}

    response = openai.ChatCompletion.create(
            model=model_name,
            messages=[messages],
    )
    raw_text = response["choices"][0]["message"]["content"]

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
        extracted_events = []
        if ("speaker_1" in raw_text) and ("speaker_2" in raw_text):
            pattern = r"(speaker_\d+:)"
            parts = re.split(pattern, raw_text)
            events = {"speaker_1": "", "speaker_2": ""}

            for i in range(1, len(parts), 2):
                speaker = parts[i].strip(':')
                event = parts[i+1].strip().rstrip(',')
                if speaker in events:
                    events[speaker] += event

            for speaker, event in events.items():
                if event:
                    extracted_events.append(f"{speaker}: {event}")
        else:
            extracted_events = raw_text
        print(extracted_events)
    return extracted_events

def read_conversations(data_path: str) -> List[str]:
    with open(data_path, "r", encoding="utf-8") as file:
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
    
def speaker_mapping(speaker):
    if speaker in ["A","Elisheva","Roper","Aneia","Alyzza","Anaiah","Amani","Ismar","Kyonna","A (Keltsey)","Aahron","Alyvia"]:
        return "speaker_1"
    elif speaker in ["B","Friend","Ryna","Nakea","Friend A","Friend B","Zacori","Aashir","C"]:
        return "speaker_2"
    raise ValueError(f"Invalid speaker: {speaker}")

def timelychat_read_conversations(data_path: str) -> List[str]:
    if args.dataset == "timelychat":
        data = load_dataset(data_path, split='eval')
    elif args.dataset == "gapchat":
        with open(data_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    conversations = []
    skip_utterance_lst = ["[B reads the message]","(Avantae resumes writing the letter after the delay)","", ".","[B is typing a message...]","[Bends knee after enduring the pain]","\"I can't believe this happened.\"]","[Awaiting B's last message]","[B leaves the chat]","[A leaves the chat]","[silence]","[B gathers the courage and confesses his love to Shae]","[end of instructions]"]
    for session in data:
        conversation = ""
        for utterance_idx in range(len(session['context'])):
            utterance = session['context'][utterance_idx]
            if utterance in skip_utterance_lst:
                continue
            speaker = speaker_mapping(session['speaker_list'][utterance_idx])
            conversation += speaker + ": " + utterance + "\n"
        conversations.append(conversation)
    print(len(conversations))
    return conversations


def read_log_data(data_path: str) -> List[Dict]:
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

def read_events(data_path: str) -> List[str]:
    with open(data_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    events = []
    for item in data:
        event = {
           f"speaker_{id}": item[f"speaker_{id}"] if f"speaker_{id}" in item and item[f"speaker_{id}"] != ["Not mentioned."] else ""
           for id in [1, 2]
        }
        # if all empty, append empty string
        if not any(event.values()):
            events.append("")
            continue
        
        events.append(json.dumps(event, indent=4))       
    return events

def extract_events(data_path: str) -> None:
    events_list = []
    if args.dataset == "gapchat":
        conversations = read_conversations(data_path)
    elif args.dataset == "timelychat":
        conversations = timelychat_read_conversations(data_path)
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
    if args.dataset == "gapchat":
        save_path = f"./new_data/{args.split}/extracted_events.json"
    elif args.dataset == "timelychat":
        save_path = f"/home/minjinj/timely-chat/resources/data/new_data/extracted_events.json"
    with open(save_path, 'a+', encoding='utf-8') as events_file:
        json.dump(events_list, events_file, indent=4)
    

def get_events_from_logs(log_path: Text):
    events_list = []
    with open(log_path, 'r', encoding='utf-8') as log_file:
        log_data = log_file.readlines()
    counter = 0
    for line in log_data:
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
    if args.dataset == "gapchat":
        save_path = f"./new_data/{args.split}/extracted_events.json"
    elif args.dataset == "timelychat":
        save_path = f"/home/minjinj/timely-chat/resources/data/new_data/extracted_events.json"
    with open(save_path, 'a+', encoding='utf-8') as events_file:
        json.dump(events_list, events_file, indent=4)


def estimate_time(event_path: str) -> None:
    counter = 0
    events = read_events(event_path)
    if args.dataset == "gapchat":
        time_tagged_file_path = f"./new_data/{args.split}/time_tag.jsonl"
    elif args.dataset == "timelychat":
        time_tagged_file_path = "/home/minjinj/timely-chat/resources/data/new_data/time_tag.jsonl"
    time_tagged_file = open(time_tagged_file_path, 'a+', encoding='utf-8')
    for event in events[counter:]:
        time_tag = {
            "speaker_1": [],
            "speaker_2": []
        }
        
        if event != "":
            print(f"Processing {counter}")
            extracted_time = get_chatgpt_response(event, task="duration")
            if type(extracted_time) == str:
                extracted_time = [extracted_time]
            for time in extracted_time:
                time_split = time.split(":")
                parts = time_split[1].split(",")
                speaker = time_split[0].replace(" ", "").lower()
                for part in parts: # one event in a specific speaker
                    if "->" in part:
                        duration = part.split("->")[1].strip()
                        time_tag[speaker].append(duration)
        time_tagged_file.write(json.dumps(time_tag) + "\n")
        counter += 1
    time_tagged_file.close()


def get_schedule(event_path: str) -> None:
    default_schedule = {"speaker_1": [], "speaker_2": []}
    events = read_events(event_path)
    counter = 0
    print(len(events))
    if args.dataset == "gapchat":
        schedule_file_path = f"./new_data/{args.split}/schedule.jsonl"
    elif args.dataset == "timelychat":
        schedule_file_path = f"/home/minjinj/timely-chat/resources/data/new_data/schedule.jsonl"
    with open(schedule_file_path, 'a+', encoding='utf-8') as event_schedule_file:
        for event in events[counter:]:
            print(json.dumps(event, indent=4))
            schedule = default_schedule
            if event and event != "Fail":
                print(f"Processing {counter}")
                schedule = get_chatgpt_response(event, task="schedule")
            if schedule == "Fail":
                schedule = default_schedule
            counter += 1
            event_schedule_file.write(json.dumps(schedule) + "\n")

def get_evaluation_conversation(save_path: str) -> None:
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
    log_file_path = "./data/new_data/self_chatgpt_log.jsonl"
    with open(log_file_path, "a+", encoding='utf-8') as log_file, open(save_path, 'a+', encoding='utf-8') as conversation_file:
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
    parser.add_argument('--dataset', type=str, default="gapchat", choices=['gapchat','timelychat'])
    args = parser.parse_args()
    print(args)
    if args.dataset == "gapchat":
        data_path = f"./gap_chat/{args.split}/merge.json" # path to gap_chat data
        events_path = f"./new_data/{args.split}/extracted_events.json" # path to events
        #read_conversations(data_path)
    elif args.dataset == "timelychat":
        #login(token="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")
        data_path = 'seongbo-research/timelychat' # path to gap_chat data
        events_path = "/home/minjinj/timely-chat/resources/data/new_data/extracted_events.json" # path to events
        #timelychat_read_conversations(data_path)
    extract_events(data_path)
    #log_path = f"./new_data/{args.split}/events_log.jsonl"
    #get_events_from_logs(log_path)
    estimate_time(events_path)
    get_schedule(events_path)

    # self_chat_gpt_path = r"./new_data/self_chat_gpt.jsonl"
    # for index in range(15):
    #     get_evaluation_conversation(self_chat_gpt_path)