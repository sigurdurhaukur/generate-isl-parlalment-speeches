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
    save_dir = "processed_data/"
    all_paths = collect_all_paths_in_dir("data/", max_paths=None)

    if os.path.exists(save_dir):
        clean_dir(save_dir)
        print(f"Cleaned {save_dir}")
    else:
        os.mkdir(save_dir)
        print(f"Created {save_dir}")

    print("Processing...")
    for i, path in enumerate(all_paths):
        try:
            data = XML_file(path).read_xml()
            new_path = save_dir + str(i) + ".txt"
            save(new_path, data)

            print(
                f"Processed:  {i} / {len(all_paths)} total left: {len(all_paths) - i}"
            )
        except Exception as e:
            print("could not process file: " + path)
            print("Error: " + str(e))

    print("Done!")
