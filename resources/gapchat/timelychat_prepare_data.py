import json
import re
from math import floor
import argparse
from typing import Dict, List, Tuple
from huggingface_hub import login
from datasets import load_dataset

def speaker_mapping(speaker):
    if speaker in ["A","Elisheva","Roper","Aneia","Alyzza","Anaiah","Amani","Ismar","Kyonna","A (Keltsey)","Aahron","Alyvia"]:
        return "speaker_1"
    elif speaker in ["B","Friend","Ryna","Nakea","Friend A","Friend B","Zacori","Aashir","C"]:
        return "speaker_2"
    raise ValueError(f"Invalid speaker: {speaker}")

def read_conversations(data_path: str):
    data = load_dataset(data_path, split='eval')
    conversations = []
    for session in data:
        conversation = ""
        for utterance_idx in range(len(session['context'])):
            utterance = session['context'][utterance_idx]
            if utterance in ["[B reads the message]","(Avantae resumes writing the letter after the delay)","", ".","[B is typing a message...]","[Bends knee after enduring the pain]","\"I can't believe this happened.\"]","[Awaiting B's last message]","[B leaves the chat]","[A leaves the chat]","[silence]","[B gathers the courage and confesses his love to Shae]","[end of instructions]"]:
                continue
            speaker = speaker_mapping(session['speaker_list'][utterance_idx])
            conversation += speaker + ": " + utterance + "\n"
        conversations.append(conversation)
    print(len(conversations))
    return conversations

                                   
def merge_utterance(conversation: Dict, session_idx: int, utterance_idx: int) -> Tuple[str, int]:
    """Merge utterances if the same speaker speaks sequentially

    Args:
        conversation (Dict) : conversation dictionary
        session_idx (int) : index of the session
        utterance_idx (int) : index of the utterance in the session

    Returns:
        text (str) : merged utterance (e.g. "hi","how are you" -> "hi how are you")
        utterance_idx (int) : current utterance index (e.g. return 2 if "hi" is utterance index 1, "how are you" is utterance index 2)
    """
    utterance = conversation['context'][utterance_idx]
    text = utterance
    speaker = conversation['speaker_list'][utterance_idx]
    while utterance_idx < len(conversation['context']) - 1:
        next_utterance = conversation['context'][utterance_idx + 1]
        next_speaker = conversation['speaker_list'][utterance_idx + 1]
        
        if speaker != next_speaker:
            break
        
        utterance_idx += 1
        text += " " + next_utterance
        
    return text, utterance_idx


# 범위 시간값에서 평균을 계산하는 함수
def average_range(range_str):
    range_values = [float(x) for x in range_str.split('-')]
    return sum(range_values) / len(range_values)

def time_to_minutes(time_str: str) -> int:
    """Transform time_str to minutes

    Args:
        time_str (Text): time string e.g. 1 hours, 38 minutes ...

    Returns:
        str: time represented minutes e.g. 1 hours -> 60
    """
    word_to_time = {
        'few': 3,      # 3 hours as few
        'several': 3,  # 7 hours as several
        'one': 1,      # 1 hour
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'ten': 10,
        'eleven': 11,
        'twelve': 12,
        'thirteen': 13,
        'fourteen': 14,
        'fifteen': 15,
        'sixteen': 16,
        'seventeen': 17,
        'eighteen': 18,
        'nineteen': 19,
        'twenty': 20,
        'thirty': 30,
        'forty': 40,
        'fifty': 50,
        'a' : 1,
        'an' : 1,
        'all':1,
        'many':8,
        'couple':2,
    }
    # extract quantity, time unit and word
    time_number_pattern = re.compile(r'(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s*(hour|hours|minute|minutes|second|seconds|week|weeks|month|months|year|years|hour|hours|days|day|minutes|minute|season|seasons|evening|morning|afternoon|semester|semesters|night|nights|daily)')
    time_word_pattern = re.compile(
        r'(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|few|several|a|an|couple|all|many)\b\s*(?:of\s*)?(hour|hours|minute|minutes|second|seconds|week|weeks|month|months|year|years|hour|hours|days|day|minutes|minute|season|seasons|evening|morning|afternoon|semester|semesters|night|nights|daily)?'
    )
    time_word_matches = time_word_pattern.findall(time_str.lower())
    time_number_matches = time_number_pattern.findall(time_str.lower())

    quantity_lst = []
    unit_lst = []

    for match in time_word_matches:
        word, unit = match
        if word and unit:
            quantity_lst.append(word_to_time[word])
            unit_lst.append(unit)

    for match in time_number_matches:
        number, unit = match
        if number and unit:
            if "-" in number:
                number = average_range(number)
            quantity_lst.append(float(number))
            unit_lst.append(unit)

    if len(quantity_lst) != 1:
        print(f"Here is a two time: {time_str}")
    quantity = quantity_lst[0]
    unit = unit_lst[0]       
    
    # transform time to minute using time basis
    if unit in ['week', 'weeks']:
        return quantity * 7 * 24 * 60 
    elif unit in ['month', 'months']:
        return quantity * 30 * 24 * 60
    elif unit in ['year', 'years']:
        return quantity * 365 * 24 * 60
    elif unit in ['hour', 'hours']:
        return quantity * 60
    elif unit in ['day', 'days','daily']:
        return quantity * 60 * 24
    elif unit in ['minute', 'minutes']:
        return quantity
    elif unit in ['season', 'seasons']:
        return quantity * 30 * 24 * 60 * 3
    elif unit in ['evening','morning','afternoon']:
        return quantity * 60 * 8
    elif unit in ['semester','semesters']:
        return quantity * 30 * 24 * 60 * 6
    elif unit in ['night', 'nights']:
        return quantity * 60 * 12
    elif unit in ['seconds','second']:
        return 0
    else:
        raise ValueError(f"Unsupported time unit: {unit}")

def convert_minutes(total_minutes: int) -> str:
    """Convert total minutes into the largest possible unit

    Args:
        total_minutes (Text): minutes

    Returns:
        str: time represented largest possible unit e.g. 60 -> 1 hour
    """
    MINUTES_PER_HOUR = 60
    HOURS_PER_DAY = 24
    DAYS_PER_WEEK = 7
    DAYS_PER_MONTH = 30 
    DAYS_PER_YEAR = 365

    minutes = total_minutes
    hours = total_minutes / MINUTES_PER_HOUR
    days = total_minutes / (MINUTES_PER_HOUR * HOURS_PER_DAY)
    weeks = days / DAYS_PER_WEEK
    months = days / DAYS_PER_MONTH
    years = days / DAYS_PER_YEAR
    
    # convert the result into the largest unit
    if years >= 1:
        return f"{floor(years)} year{'s' if floor(years) > 1 else ''}"
    elif months >= 1:
        return f"{floor(months)} month{'s' if floor(months) > 1 else ''}"
    elif weeks >= 1:
        return f"{floor(weeks)} week{'s' if floor(weeks) > 1 else ''}"
    elif days >= 1:
        return f"{floor(days)} day{'s' if floor(days) > 1 else ''}"
    elif hours >= 1:
        return f"{floor(hours)} hour{'s' if floor(hours) > 1 else ''}"
    else:
        return f"{total_minutes} minute{'s' if floor(minutes) > 1 else ''}"

def sum_time_units(time_list: List[str]) -> str:
    """Takes a list of time strings, calculates the total, and converts it into the largest possible unit

    Args:
        time_list (List): list of time string

    Returns:
        str: time string converted into the largest possible unit (e.g. [8 hours, 17 hours, 1 minutes] -> 1 day)
    """
    total_minutes = sum(time_to_minutes(time) for time in time_list)
    return convert_minutes(total_minutes)


def get_progress_label(gaps: str, duration: str) -> str:
    """Generate progress label

    Args:
        gaps (Text): Input dialogue time gap
        duration (Text): Input event duration

    Returns:
        str: label in ["Finished","Three-forth Finished", "Half Finished", "One-forth Finished","No significant progress"].
    """
    duration = duration.strip() #remove white space

    if gaps == "0 s" or "N/A" in duration:
        return "No significant progress."
        
    if "indefinite" in duration:
        duration = "2 years"
    if "ongoing" in duration:
        duration = "1 day"
        
    gap_time = time_to_minutes(gaps)
    duration_time = time_to_minutes(duration)
    if gap_time >= duration_time:
        return "Finished."
    elif gap_time >= 0.75 * duration_time:
        return "Three-forth Finished."
    elif gap_time >= 0.5 * duration_time:
        return "Half Finished."
    elif gap_time >= 0.25 * duration_time:
        return "One-forth Finished."
    return "No significant progress"

def gap_time_cleaning(gap:str):
    if "it took" in gap:
        return gap[8:]
    elif "instant" in gap:
        return "0 minutes"
    return gap

def prepare_time_aware_data(conversation_path: str, events_path: str, time_tag_path: str, schedule_path: str) -> None:
    conversation_data = load_dataset(conversation_path,split='eval')
    with open(events_path, 'r', encoding='utf-8') as events_file:
        events_data = json.load(events_file)
    with open(time_tag_path, 'r', encoding='utf-8') as time_file, open(schedule_path, 'r', encoding='utf-8') as schedule_file:
        time_tag_data = time_file.readlines()
        schedule_data = schedule_file.readlines()
        
    # check if the data is valid
    session_number = len(conversation_data)
    assert(session_number == len(events_data)) # extracted event, time_tag based on each session
    assert(session_number == len(time_tag_data))
    assert(session_number == len(schedule_data))

    # check if the number of event is equal to the number of time tag
    for index in range(session_number):
        tag_data = json.loads(time_tag_data[index])
        for speaker in ['speaker_1','speaker_2']:
            event_length = len(events_data[index][speaker])
            if "Not mentioned." in events_data[index][speaker]:    
                event_length = 0
            time_tag_length = len(tag_data[speaker])
            try:
                assert(event_length == time_tag_length)
            except:
                print(index)
    
    # convert_data
    event_text_dict = {"speaker_1":"","speaker_2":"",}
    finished_text_dict = {"speaker_1":"","speaker_2":"",}
    to_do_text_dict = {"speaker_1":"","speaker_2":"",}
    event_text = ""
    to_do_text = ""
    finished_text = ""
    gap = ""
    skip_utt_lst = ["[B reads the message]","(Avantae resumes writing the letter after the delay)","", ".","[B is typing a message...]","[Bends knee after enduring the pain]","\"I can't believe this happened.\"]","[Awaiting B's last message]","[B leaves the chat]","[A leaves the chat]","[silence]","[B gathers the courage and confesses his love to Shae]","[end of instructions]"]
    data_dict_lst = []
    for data_idx in range(len(conversation_data)):
        conversation = conversation_data[data_idx]
        session_length = 1
        if  "duration}" in conversation['time_elapsed']:
            continue
        gap_lst = [gap_time_cleaning(conversation['time_elapsed'])]
        for session_idx in range(session_length):            
            ### handle extracted events and generate progress label ###
            # During the first session, event and to_do are mapped with initial progress in the data
            for speaker in ["speaker_1","speaker_2"]:
            ### generate progress label ###
                event_lst = events_data[data_idx][speaker]
                for e_idx in range(len(event_lst)):
                    if event_lst[e_idx] != "Not mentioned.":
                        gap = gap_lst[0]
                        tag_info = json.loads(time_tag_data[data_idx])
                        #if data_idx == 4:
                        #   breakpoint()
                        progress_label = get_progress_label(gap, tag_info[speaker][e_idx])
                        event_text += f"{event_lst[e_idx]} [{progress_label}], "
                        print(f"data_session_idx: {data_idx}-{session_idx}-{speaker}, event_cnt: {data_idx+1} \nevent: {event_lst[e_idx]}, time_gap: {gap}, tag_info: {tag_info[speaker][e_idx]}, progress_label: {progress_label}")
                event_text_dict[speaker] = event_text[:-2]
                
                ### generate schedule label ###
                schedule_lst_text = json.loads(schedule_data[data_idx])[speaker]
                try:
                    # just pass if there is not schedule
                    if (len(schedule_lst_text)==0) or ("No event" in schedule_lst_text[0]) or ("No schedule" in schedule_lst_text[0]) or ("Not enough information" in schedule_lst_text[0]) or ("Fail" in schedule_lst_text[0]):
                        continue
                    for s in schedule_lst_text: # for each event
                        step_lst = s.split(", ")
                        time_pattern = re.compile(r'\b\d+\s*(?:week|weeks|day|days|hour|hours|month|months|minute|minutes|years|year|semester|semesters)\b', re.IGNORECASE)
                        parentheses_pattern = re.compile(r'\(.*?\)', re.IGNORECASE)
                        step_time_info = []
                        for task in step_lst:
                            task = re.sub(parentheses_pattern, '', task).strip()
                            for match in time_pattern.finditer(task):
                                step_time_info.append(match.group())
                        for step_idx in range(len(step_time_info)): # for each step
                            cumulated_step_time = sum_time_units(step_time_info[:step_idx+1])
                            progress_label = get_progress_label(gap, cumulated_step_time)
                            if progress_label == "Finished.":
                                finished_text += step_lst[step_idx]+", "
                            else:
                                to_do_text += step_lst[step_idx]+", "
                    finished_text_dict[speaker] = finished_text[:-2]
                    to_do_text_dict[speaker] = to_do_text[:-2]
                    print(f"finished: {finished_text[:-2]}\nto_do: {to_do_text[:-2]}")  
                                
                
                except:
                    print("Fail to extract schedule: ",data_idx, schedule_lst_text)
            event_text = ""
            finished_text = ""
            to_do_text = ""
                                
            ### extract utterances in the session ###
            machine_speaker = speaker_mapping(conversation['target_speaker'])
            human_speaker = "speaker_2" if machine_speaker == "speaker_1" else "speaker_1"
            utterance_length = len(conversation['context'])
            dialogue_text = ""
            i = 0
            while i < utterance_length:
                # make an utterance list for each human and machine.
                if conversation["speaker_list"][i] == "  ":
                    i += 1
                    continue
                if conversation['context'][i] in skip_utt_lst:
                    i += 1
                    continue
                cur_speaker = speaker_mapping(conversation["speaker_list"][i])
                if cur_speaker not in ["speaker_1", "speaker_2"]:
                    i += 1
                    continue
                utterance_text, i = merge_utterance(conversation, session_idx, i)
                speaker = "speaker_1" if cur_speaker == human_speaker else "speaker_2"
                dialogue_text += f"<spk> {speaker}: <utt> {utterance_text} "
                i += 1

            dialogue_text = dialogue_text.strip()
            label = conversation['timely_response']
            progress_text = f"speaker_1: {event_text_dict[human_speaker]} speaker_2: {event_text_dict[machine_speaker]}"
            schedule_text = f"speaker_1: finished: {finished_text_dict[human_speaker]} to-do: {to_do_text_dict[human_speaker]} speaker_2: finished: {finished_text_dict[machine_speaker]} to-do: {to_do_text_dict[machine_speaker]}"
            
            data_dict = {"text":dialogue_text,"Progress":progress_text,"Schedule":schedule_text,"labels":label}
            
            data_dict_lst.append(data_dict)
                
            event_text_dict = {"speaker_1":"","speaker_2":"",}
            finished_text_dict = {"speaker_1":"","speaker_2":"",}
            to_do_text_dict = {"speaker_1":"","speaker_2":"",}

    with open(f"/home/minjinj/timely-chat/resources/data/new_data/both.json", 'w', encoding='utf-8') as output_file:
        json.dump(data_dict_lst,output_file)
    


def main():
    conversation_path = 'seongbo-research/timelychat'
    events_path = f"/home/minjinj/timely-chat/resources/data/new_data/extracted_events.json" # path to events
    time_tag_path = f"/home/minjinj/timely-chat/resources/data/new_data/time_tag.jsonl" # path to progress label
    schedule_path = f"/home/minjinj/timely-chat/resources/data/new_data/schedule.jsonl" # path to schedules
    prepare_time_aware_data(
        conversation_path, events_path, time_tag_path, schedule_path
    )

if __name__ == "__main__":
    main()