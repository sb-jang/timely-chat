
import json
import re
from math import floor
import argparse
from typing import Dict, List, Tuple

def prepare_time_aware_data(args, conversation_path: str, events_path: str, time_tag_path: str, schedule_path: str) -> None:
    with open(conversation_path, 'r', encoding='utf-8') as conversation_file, open(events_path, 'r', encoding='utf-8') as events_file:
        conversation_data = json.load(conversation_file)
        events_data = json.load(events_file)
    with open(time_tag_path, 'r', encoding='utf-8') as time_file, open(schedule_path, 'r', encoding='utf-8') as schedule_file:
        time_tag_data = time_file.readlines()
        schedule_data = schedule_file.readlines()
        
    parlai_format_file = open(f"./new_data/{args.split}/time.txt", 'w', encoding='utf-8')
    parlai_schedule_format_file = open(f"./new_data/{args.split}/schedule.txt", 'w', encoding='utf-8')
    parlai_both_format_file = open(f"./new_data/{args.split}/both.txt", 'w', encoding='utf-8')    
    
    # check if the data is valid
    session_number = 0
    for item in conversation_data:
        session_number += len(item['sessions'])
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
    event_cnt = 0
    for data_idx in range(len(conversation_data)):
        conversation = conversation_data[data_idx]
        session_length = len(conversation['sessions'])
        gap_lst = [conversation['gap'][idx] for idx in range(len(conversation['gap'])) if idx%2==0]
        for session_idx in range(session_length):            
            ### handle extracted events and generate progress label ###
            # During the first session, event and to_do are mapped with initial progress in the data
            for speaker in ["speaker_1","speaker_2"]:
                if session_idx == 0:
                        event_text_dict[speaker] = f"{conversation['initial'][speaker]['progress']}"
                        to_do_text_dict[speaker] = f"{conversation['initial'][speaker]['progress']}"
                        gap = ""
                else:
                    ### generate progress label ###
                    event_lst = events_data[event_cnt][speaker]
                    for e_idx in range(len(event_lst)):
                        if event_lst[e_idx] != "Not mentioned.":
                            gap = gap_lst[session_idx-1]
                            tag_info = json.loads(time_tag_data[event_cnt])
                            progress_label = get_progress_label(gap, tag_info[speaker][e_idx])
                            event_text += f"{event_lst[e_idx]} [{progress_label}], "
                            print(f"data_session_idx: {data_idx}-{session_idx}-{speaker}, event_cnt: {event_cnt+1} \nevent: {event_lst[e_idx]}, time_gap: {gap}, tag_info: {tag_info[speaker][e_idx]}, progress_label: {progress_label}")
                    event_text_dict[speaker] = event_text[:-2]
                    
                    ### generate schedule label ###
                    schedule_lst_text = json.loads(schedule_data[event_cnt])[speaker]
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
                        print("Fail to extract schedule: ",event_cnt, schedule_lst_text)
                event_text = ""
                finished_text = ""
                to_do_text = ""
                                
            ### extract utterances in the session ###
            for human_speaker, machine_speaker in [("speaker_1", "speaker_2"), ("speaker_2", "speaker_1")]:
            # create 2 data per session: one with speaker_1 as human, one with speaker_2 as human
                utterance_length = len(conversation["sessions"][session_idx])
                human_lst = []
                machine_lst = []
                i = 0
                while i < utterance_length:
                    # make an utterance list for each human and machine.
                    cur_speaker = conversation["sessions"][session_idx][i]["speaker"]
                    if cur_speaker not in ["speaker_1", "speaker_2"]:
                        i += 1
                        continue
                    utterance_text, i = merge_utterance(conversation, session_idx, i)
                    if cur_speaker == human_speaker:
                            human_lst.append(utterance_text)
                    else:
                        machine_lst.append(utterance_text)
                        if len(human_lst) == 0:
                            human_lst.append("")
                    i += 1

                n_turn = min(len(human_lst),len(machine_lst))
                for i in range(n_turn):
                    utterance_text = human_lst[i]
                    label_text = machine_lst[i]
                    gap_text = "No Gap"
                    if gap != "":
                        gap_text = gap
                    
                    suffix = "\tepisode_done:True" if i == n_turn - 1 else ""
                    suffix += "\tfinal_session:True" if i == n_turn - 1 and session_idx == session_length-1 else ""
                    parlai_format_file.write(f"text:{utterance_text}\\n Progress:speaker_1: {event_text_dict[human_speaker]} speaker_2: {event_text_dict[machine_speaker]}\\n Gap:{gap_text}\tlabels:{label_text}{suffix}\n")
                    parlai_schedule_format_file.write(f"text:{utterance_text}\\n Schedule:speaker_1: finished: {finished_text_dict[human_speaker]} to-do: {to_do_text_dict[human_speaker]} speaker_2: finished: {finished_text_dict[machine_speaker]} to-do: {to_do_text_dict[machine_speaker]}\\n Gap:{gap_text}\tlabels:{label_text}{suffix}\n")
                    parlai_both_format_file.write(f"text:{utterance_text}\\n Progress:speaker_1: {event_text_dict[human_speaker]} speaker_2: {event_text_dict[machine_speaker]}\\n Schedule:speaker_1: finished: {finished_text_dict[human_speaker]} to-do: {to_do_text_dict[human_speaker]} speaker_2: finished: {finished_text_dict[machine_speaker]} to-do: {to_do_text_dict[machine_speaker]}\\n Gap:{gap_text}\tlabels:{label_text}{suffix}\n")
                        
            event_text_dict = {"speaker_1":"","speaker_2":"",}
            finished_text_dict = {"speaker_1":"","speaker_2":"",}
            to_do_text_dict = {"speaker_1":"","speaker_2":"",}
                        
            if session_idx != 0:
                            event_cnt += 2 if session_idx == (session_length - 1) else 1    
                            
    parlai_format_file.close()
    parlai_schedule_format_file.close()
    parlai_both_format_file.close()
                                   
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
    session = conversation["sessions"][session_idx]
    utterance = session[utterance_idx]
    
    text = utterance["text"].replace("\n", " ").strip()
    speaker = utterance["speaker"]
    while utterance_idx < len(session) - 1:
        next_utterance = session[utterance_idx + 1]
        next_speaker = next_utterance["speaker"]
        
        if speaker != next_speaker:
            break
            
        utterance_idx += 1
        text += " " + next_utterance["text"].replace("\n", " ").strip()
        
    return text, utterance_idx


def time_to_minutes(time_str: str) -> int:
    """Transform time_str to minutes

    Args:
        time_str (Text): time string e.g. 1 hours, 38 minutes ...

    Returns:
        str: time represented minutes e.g. 1 hours -> 60
    """
    # extract quantity and time unit
    match = re.match(r'(\d+)\s*(week|weeks|month|months|year|years|hour|hours|days|day|minutes|minute|season|seasons|evening|morning|afternoon|semester|semesters|night|nights|daily)', time_str, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid time format: {time_str}")
    quantity = int(match.group(1))
    unit = match.group(2).lower()
    
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


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--split', type=str, default="train", choices=['train','eval'])
    
    args = parser.parse_args()
    
    conversation_path = f"./gap_chat/{args.split}/merge.json" # path to gap_chat data
    events_path = f"./new_data/{args.split}/extracted_events.json" # path to events
    time_tag_path = f"./new_data/{args.split}/time_tag.jsonl" # path to progress label
    schedule_path = f"./new_data/{args.split}/schedule.jsonl" # path to schedules
    prepare_time_aware_data(
        args, conversation_path, events_path, time_tag_path, schedule_path
    )

if __name__ == "__main__":
    main()
