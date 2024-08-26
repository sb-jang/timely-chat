
import json
import re
from math import floor
import argparse

def prepare_time_aware_data(args, conversation_path, events_path, time_tag_path, schedule_path):
    conversation_file = open(conversation_path, 'r', encoding='utf-8')
    events_file = open(events_path, 'r', encoding='utf-8')
    time_file = open(time_tag_path, 'r', encoding='utf-8')
    schedule_file = open(schedule_path, 'r', encoding='utf-8')


    parlai_format_file = open(f"./new_data/{args.path}/time.txt", 'w', encoding='utf-8')
    parlai_schedule_format_file = open(f"./new_data/{args.path}/schedule.txt", 'w', encoding='utf-8')
    parlai_both_format_file = open(f"./new_data/{args.path}/both.txt", 'w', encoding='utf-8')
    # valid data
    conversation_data = json.load(conversation_file)
    events_data = json.load(events_file)
    time_tag_data = time_file.readlines()
    schedule_data = schedule_file.readlines()

    session_number = 0
    for item in conversation_data:
        session_number += len(item['sessions'])
    assert(session_number == len(events_data)) # extracted event, time_tag based on each session
    assert(session_number == len(time_tag_data))
    assert(session_number == len(schedule_data))
            
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
    event_text = ""
    finished_text = ""
    to_do_text = ""
    event_cnt = 0
    for data_idx in range(len(conversation_data)):
        conversation = conversation_data[data_idx]
        session_length = len(conversation['sessions'])
        gap_lst = [conversation['gap'][idx] for idx in range(len(conversation['gap'])) if idx%2==0]
        for session_idx in range(session_length):
            for speaker in ['speaker_1','speaker_2']:
                # handle extracted events and generate progress label
                if session_idx == 0:
                        event_text = f"{conversation['initial'][speaker]['progress']}"
                        to_do_text = f"{conversation['initial'][speaker]['progress']}"
                else:
                    ### generate progress label ###
                    event_lst = events_data[event_cnt][speaker]
                    for e_idx in range(len(event_lst)):
                        if event_lst[e_idx] != "Not mentioned.":
                            gap = gap_lst[session_idx-1]
                            tag_info = json.loads(time_tag_data[event_cnt])
                            #print(f"Loading from time tag at {event_cnt}, for speaker_1, event: {e_idx}")
                            progress_label = get_progress_label(gap, tag_info[speaker][e_idx])
                            event_text += f"{event_lst[e_idx]} [{progress_label}], "
                            print(f"data_session_idx: {data_idx}-{session_idx}-{speaker}, event_cnt: {event_cnt+1} \nevent: {event_lst[e_idx]}, time_gap: {gap}, tag_info: {tag_info[speaker][e_idx]}, progress_label: {progress_label}")
                    event_text = event_text[:-2]
                    
                    ### generate schedule label ###
                    if "Fail" in schedule_data[event_cnt]:
                        schedule_lst_text = ["Fail"]
                    else:
                        schedule_lst_text = json.loads(schedule_data[event_cnt])[speaker]
                    try:
                        if ("No event" in schedule_lst_text[0]) or ("No schedule" in schedule_lst_text[0]) or ("Not enough information" in schedule_lst_text[0]) or ("Fail" in schedule_lst_text[0]):
                            pass
                        else:
                            for s in schedule_lst_text: # for each event
                                step_lst = s.split(", ")
                                time_pattern = re.compile(r'\b\d+\s*(?:week|weeks|day|days|hour|hours|month|months|minute|minutes|years|year|semester|semesters)\b', re.IGNORECASE)
                                step_time_info = [match.group() for task in step_lst for match in time_pattern.finditer(task)]
                                for step_idx in range(len(step_time_info)): # for each step
                                    cumulated_step_time = sum_time_units(step_time_info[:step_idx+1])
                                    progress_label = get_progress_label(gap, cumulated_step_time)
                                    if progress_label == "Finished.":
                                        try:
                                            finished_text += step_lst[step_idx]+", "
                                        except:
                                            print(step_idx, len(step_lst)) #if schedule contains several time word, just skip
                                    else:
                                        try:
                                            to_do_text += step_lst[step_idx]+", "
                                        except:
                                            print(step_idx, len(step_lst))
                            finished_text = finished_text[:-2]
                            to_do_text = to_do_text[:-2]
                            print(f"finished: {finished_text}\nto_do: {to_do_text}")            
                    
                    except:
                        print(event_cnt, schedule_lst_text)

                        
                utterance_length = len(conversation["sessions"][session_idx])
                human_lst = []
                machine_lst = []
                i = 0
                while i < utterance_length:
                    if conversation["sessions"][session_idx][i]['speaker'] in ["speaker_1", "speaker_2"]:
                        if (len(human_lst) == 0) and (conversation["sessions"][session_idx][i]['speaker'] != speaker):
                            human_lst.append("")
                            utterance_text, i = merge_utterance(conversation, session_idx, i) 
                            machine_lst.append(utterance_text)
                        else:
                            if conversation["sessions"][session_idx][i]['speaker'] == speaker:
                                utterance_text, i = merge_utterance(conversation, session_idx, i)  
                                human_lst.append(utterance_text)
                            else:
                                utterance_text, i = merge_utterance(conversation, session_idx, i)  
                                machine_lst.append(utterance_text)
                        i += 1
                    else:
                        i += 1
            
                n_turn = min(len(human_lst),len(machine_lst))
                for i in range(n_turn):
                    utterance_text = human_lst[i]
                    label_text = machine_lst[i]
                    
                    if i == (n_turn -1):
                        parlai_format_file.write(
                                "text:" + utterance_text + "\\n Progress: " + event_text + "\t" + "labels:" + label_text + "\t" + "episode_done:True\n"
                        )
                        parlai_schedule_format_file.write(
                            "text:" + utterance_text + "\\n Schedule: finished: " + finished_text +" to-do: "+ to_do_text + "\t" + "labels:" + label_text + "\t" + "episode_done:True\n"
                        )
                        parlai_both_format_file.write(
                            "text:" + utterance_text + "\\n Progress: " + event_text + "\\n Schedule: finished:" + finished_text +" to-do:"+ to_do_text + "\t" + "labels:" + label_text + "\t" + "episode_done:True\n"
                        )

                    else:
                        parlai_format_file.write(
                        "text:" + utterance_text + "\\n Progress:" + event_text + "\t" + "labels:" + label_text + "\n"
                        )
                        parlai_schedule_format_file.write(
                        "text:" + utterance_text +  "\\n Schedule: finished:" + finished_text +" to-do:"+ to_do_text + "\t" + "labels:" + label_text + "\n"
                        )
                        parlai_both_format_file.write(
                        "text:" + utterance_text + "\\n Progress: " + event_text + "\\n Schedule: finished:" + finished_text +" to-do:"+ to_do_text + "\t" + "labels:" + label_text + "\n"
                        )
                        
                event_text = ""
                finished_text = ""
                to_do_text = ""
            
            
            if session_idx == 0:
                pass
            elif session_idx == (session_length-1): # skip extracted events in last session
                event_cnt += 2
            else:
                event_cnt += 1
            
def merge_utterance(conversation, session_idx, utterance_idx):
    text = conversation["sessions"][session_idx][utterance_idx]['text'].replace("\n"," ").strip()
    speaker = conversation["sessions"][session_idx][utterance_idx]['speaker']
    try:
        next_speaker = conversation["sessions"][session_idx][utterance_idx+1]['speaker']
    except:
        return text, utterance_idx
    while speaker == next_speaker:
        utterance_idx +=1
        text += " "+conversation["sessions"][session_idx][utterance_idx]['text'].replace("\n"," ").strip()
        speaker = conversation["sessions"][session_idx][utterance_idx]['speaker']
        if utterance_idx == len(conversation["sessions"][session_idx])-1:
            return text, utterance_idx
        next_speaker = conversation["sessions"][session_idx][utterance_idx+1]['speaker']
    return text, utterance_idx

def time_to_minutes(time) -> int:
    """Convert the time to minutes.

    Args:
        time (Text): Input time.

    Returns:
        int: Time in minutes.
    """
    if time == "0 s":
        return 0
    time_parts = time.split(" ")
    if "-" in time_parts[0]:
        min = int(time_parts[0].split('-')[0])
        max = int(time_parts[0].split('-')[1])
        time_parts[0] = int((min + max) / 2)

    time_number = int(time_parts[0])
    time_unit = time_parts[1]
    assert (
        "minute" in time_unit
        or "hour" in time_unit
        or "day" in time_unit
        or "week" in time_unit
        or "month" in time_unit
        or "year" in time_unit
    )
    multiplyer = 1
    if "hour" in time_unit:
        multiplyer *= 60
    elif "day" in time_unit:
        multiplyer = multiplyer * 60 * 24
    elif "week" in time_unit:
        multiplyer = multiplyer * 60 * 24 * 7
    elif "month" in time_unit:
        multiplyer = multiplyer * 60 * 24 * 30
    elif "year" in time_unit:
        multiplyer = multiplyer * 60 * 24 * 365

    return time_number * multiplyer

def time_to_minutes(time_str):
    """주어진 시간 문자열을 분으로 변환합니다."""
    # 정규 표현식을 사용하여 숫자와 시간 단위를 추출합니다.
    match = re.match(r'(\d+)\s*(week|weeks|month|months|year|years|hour|hours|days|day|minutes|minute|season|seasons|evening|morning|afternoon|semester|semesters|night|nights|daily)', time_str, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid time format: {time_str}")
    quantity = int(match.group(1))
    unit = match.group(2).lower()
    
    # 시간 단위에 따라 분으로 변환
    if unit in ['week', 'weeks']:
        return quantity * 7 * 24 * 60  # 주를 분으로 변환
    elif unit in ['month', 'months']:
        return quantity * 30 * 24 * 60  # 월을 분으로 변환 (30일 기준)
    elif unit in ['year', 'years']:
        return quantity * 365 * 24 * 60  # 년을 분으로 변환 (365일 기준)
    elif unit in ['hour', 'hours']:
        return quantity * 60  # 시간을 분으로 변환
    elif unit in ['day', 'days','daily']:
        return quantity * 60 * 24  # 하루를 분으로 변환
    elif unit in ['minute', 'minutes']:
        return quantity  # 분을 분으로 변환
    elif unit in ['season', 'seasons']:
        return quantity * 30 * 24 * 60 * 3
    elif unit in ['evening','morning','afternoon']:
        return quantity * 60 * 8
    elif unit in ['semester','semesters']:
        return quantity * 30 * 24 * 60 * 6
    elif unit in ['night', 'nights']:
        return quantity * 60 * 12  # 밤(12시간)을 분으로 변환
    else:
        raise ValueError(f"Unsupported time unit: {unit}")

def convert_minutes(total_minutes):
    """총 분을 가장 큰 단위로 변환합니다."""
    MINUTES_PER_HOUR = 60
    HOURS_PER_DAY = 24
    DAYS_PER_WEEK = 7
    DAYS_PER_MONTH = 30  # 월을 30일로 가정
    DAYS_PER_YEAR = 365  # 년을 365일로 가정

    hours = total_minutes / MINUTES_PER_HOUR
    days = total_minutes / (MINUTES_PER_HOUR * HOURS_PER_DAY)
    weeks = days / DAYS_PER_WEEK
    months = days / DAYS_PER_MONTH
    years = days / DAYS_PER_YEAR
    
    # 결과를 가장 큰 단위로 변환
    if years >= 1:
        return f"{floor(years)} year"
    elif months >= 1:
        return f"{floor(months)} month"
    elif weeks >= 1:
        return f"{floor(weeks)} week"
    elif days >= 1:
        return f"{floor(days)} day"
    elif hours >= 1:
        return f"{floor(hours)} hour"
    else:
        return f"{total_minutes} minute"

def sum_time_units(time_list):
    """시간 문자열 리스트를 받아 총합을 계산하고 가장 큰 단위로 변환합니다."""
    total_minutes = sum(time_to_minutes(time) for time in time_list)
    return convert_minutes(total_minutes)


def get_progress_label(gaps, duration):
    """Generate progress label.

    Args:
        gaps (Text): Input dialogue time gap
        duration (Text): Input event duration

    Returns:
        str: label in ["Finished","Three-forth Finished", "Half Finished", "One-forth Finished","No significant progress"].
    """
    duration = duration.strip() #remove white space
    if "indefinite" in duration:
        duration = "2 years"
    if "N/A" in duration:
        return "No significant progress."
    if "ongoing" in duration:
        duration = "1 day"

    if gaps == "0 s":
        return "No significant progress."
    else:
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
        else:
            return "No significant progress"


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path', type=str, default="train", choices=['train','eval'])
    
    args = parser.parse_args()
    
    conversation_path = f"./gap_chat/{args.path}/merge.json" # path to gap_chat data
    events_path = f"./new_data/{args.path}/extracted_events.json" # path to events
    time_tag_path = f"./new_data/{args.path}/time_tag.jsonl" # path to progress label
    schedule_path = f"./new_data/{args.path}/schedule.jsonl" # path to schedules
    prepare_time_aware_data(
        args, conversation_path, events_path, time_tag_path, schedule_path
    )
    # print(time_to_minutes("1-2 hours"))

if __name__ == "__main__":
    main()
