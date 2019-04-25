import json
import pandas as pd
import atexit

annotations = []


def exit_handler():
    print("Exit: Writing to file...")
    with open(filename + '.annotated','w') as outfile:
        for data in annotations:
            outfile.write(data[0] + ' ' + data[1] + ' ' + data[2] + '\n')
        outfile.close()


if __name__ == '__main__':
    filename = "tweet_data/trump10000since2016.json"
    with open(filename,'r') as file:
        lines = file.read()
        tweets = json.loads(lines)

        atexit.register(exit_handler)
        test10 = 0

        tweets = tweets[84:]

        for index,tweet in enumerate(tweets):
            if "trump" in tweet['text'].lower():
                print("|| On tweet number: " + str(index) + " ||")
                print(tweet['text'] + "\n|| Pos or Neg or Objective? (p/n/o)")
                isValid = False

                # if test10 > 10:
                #     break

                def validate_res(response):
                    if len(response) > 0:
                        return response.lower()[0] == 'p' or response.lower()[0] == 'n' or response.lower()[0] == 'o'
                    else:
                        return False

                while(not isValid):
                    response = input()
                    isValid = validate_res(response)
                if(response.lower()[0] != 'o'):
                    annotations.append((tweet['text'].replace('\n',' '),tweet['timestamp'],response.upper()[0]))
                test10 = test10 + 1




    x = 0