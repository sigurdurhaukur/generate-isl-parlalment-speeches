import os
import xml.etree.ElementTree as ET
import re


class XML_file:
    def __init__(self, path):
        self.path = path
        self.tree = ET.parse(path)
        self.root = self.tree.getroot()

    def read_xml(self):
        try:
            text = []
            for elem in self.root.iter():
                if elem.text is not None:
                    text.append(elem.text.strip())

            extracted_text = " ".join(text)
            # remove prescript
            # extracted_text = extracted_text[1030:].strip()
            prescript_len = len(extracted_text.split("      ")[0])
            extracted_text = extracted_text[prescript_len:].strip()

            return extracted_text
        except Exception as e:
            print("Error: " + str(e))
            return ""


def collect_all_paths_in_dir(path, max_paths=None):
    all_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".xml"):
                all_paths.append(os.path.join(root, file))

            if max_paths is not None and len(all_paths) == max_paths:
                break

    if max_paths is None:
        return all_paths
    else:
        return all_paths[:max_paths]


def save(path, data):
    with open(path, "w+") as f:
        f.write(data)


def clean_dir(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            os.remove(os.path.join(root, file))


if __name__ == "__main__":
    print("Collecting all paths...")
    all_paths = collect_all_paths_in_dir("data/")

    if os.path.exists("processed_data_journals/"):
        clean_dir("processed_data_journals/")
        print("Cleaned processed_data_journals/")
    else:
        os.mkdir("processed_data_journals/")
        print("Created processed_data_journals/")

    print("Processing...")
    for i, path in enumerate(all_paths):
        try:
            data = XML_file(path).read_xml()
            new_path = "./processed_data_journals/" + str(i) + ".txt"
            save(new_path, data)

            print("Processed: " + str(i) + "/" + str(len(all_paths)))
        except Exception as e:
            print("could not process file: " + path)
            print("Error: " + str(e))

    print("Done!")
