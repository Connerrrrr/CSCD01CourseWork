# Selected Bugs for Assignment 3

## Table of Content

- [#19520](https://github.com/UTSCCSCD01/course-project-apple_team/tree/master/a3#fix-feature_name-referenced-before-assignment)
- [#19269](https://github.com/UTSCCSCD01/course-project-apple_team/tree/master/a3#sklearndatasetsload_files-select-file-extension)
- [Work Log](https://github.com/UTSCCSCD01/course-project-apple_team/tree/master/a3#work-log)

In Assignment 3, we added 1 new feature #19269 from selected 2 issues. Fixed solution and investigation can be found below each issue section.

## FIX 'feature_name' referenced before assignment

[Link](https://github.com/scikit-learn/scikit-learn/issues/19520) to the issue page.

- Investigation

- Design

- Interactions

## sklearn.datasets.load_files select file extension

[Link](https://github.com/scikit-learn/scikit-learn/issues/19269) to the issue page.

- Investigation

When using load\_files in a directory where there are different kinds of files (.txt, .png, ...), the user might want to load only certain files (*.txt for example). This feature would put load\_files closer to the function index\_directory from tensorflow.python.keras.preprocessing.dataset\_utils.py.
For MacOs users, .DStore files also get loaded which is an undesired behaviour.

- Design and Interactions

  ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a3/Images/19269-UML.png "UML")

  Above is the UML diagram for Datasets Module in sklearn. It’s sufficient to just modify the \_base.py file in particular the load\_files() function to implement the new feature. We are going to:

  - Modify the load_files() function.
  - Add a check_formats() function as a helper function for load_files().
  - Add user guide.
  - Do Acceptance testing.
  - Do Unit testing.

- Implementation

  - Original Code
  
    ```python
    @_deprecate_positional_args
    def load_files(container_path, *, description=None, categories=None,
               load_content=True, shuffle=True, encoding=None,
               decode_error='strict', random_state=0):
        ...

        for label, folder in enumerate(folders):
            target_names.append(folder)
            folder_path = join(container_path, folder)
            documents = [join(folder_path, d)
                        for d in sorted(listdir(folder_path))]
            target.extend(len(documents) * [label])
            filenames.extend(documents)

        ...
    ```

  - Code changed
  
    ```python
    def load_files(container_path, *, description=None, categories=None,
               load_content=True, shuffle=True, encoding=None,
               decode_error='strict', random_state=0, formats=None):
        ...

        for label, folder in enumerate(folders):
            target_names.append(folder)
            folder_path = join(container_path, folder)
            documents = [join(folder_path, d)
                        for d in sorted(listdir(folder_path)) if check_formats(d, formats)]
            target.extend(len(documents) * [label])
            filenames.extend(documents)

        ...
    ```

  - Code added

    ```python
    def check_formats(d, formats):
        """Helper function for load_files. 

        Return True if the document d satisfies the required formats, otherwise false.

        Parameters:
        ----------
        d: str
            document name.
        formats : list of str or None
            list of acceptable file formats. E.g. [".txt", ".pdf", ".png"]
        """

        if formats == None:
            return True
        
        for f in formats:
            if (d.endswith(f)):
                return True
        
        return False
    ```

- Testing

  - Acceptance Testing

    1. Open a python shell
    2. Type “import numpy as np”
    3. Type “from sklearn.utils import Bunch”
    4. Type “from sklearn.datasets import load_files”
    5. Type “load_files(file container path, formats=[‘.txt’, ‘.png’ (any format you like)])”
    6. The shell will list the files with the suffix txt or png indicating the txt files and png files are loaded
    7. Create a python file, repeat step 2 and 5, and print the result of load_files give the same result

    Examples:

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a3/Images/19269-acc-test-1.png "Acceptance Test")

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a3/Images/19269-acc-test-2.png "Acceptance Test")

  - Regression Test

    Acceptance test passed 100% with the test case defined in:

    /sklearn/datasets/tests/test_base.py

    Thus, the system still meets the requirement specifications.

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a3/Images/19269-reg-test.png "Regression Test")

  - Unit Testing

    Added three new unit test cases for load_files(), and the test passed successfully.

    Test defined in:

    /sklearn/datasets/tests/test_load_files.py.

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a3/Images/19269-unit-test.png "Unit Test")

- User Guide

  - Parameter’s description

    The load\_files() function is now added a new parameter - formats:

    Line 197 ~ 201 is the new added section for the parameter’s description.

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a3/Images/19269-user-guide.png "Parameter’s description")

  - Test Env SetUp

    In my Desktop, I have created a Container folder for testing purposes.

    It has below structure.

    - Container:
      - C1:
        - p1.png
        - p2.png
        - a0-1.pdf
        - t1.txt
        - (.DS_Store)
      - C2:
        - p3.png
        - p4.png
        - a0-2.pdf
        - t2.txt
        - t3.txt
        - (.DS_Store)
      - C3:
        - p5.png
        - p6.png
        - t4.txt
        - t5.txt
        - t6.txt
        - (.DS_Store)

    The structure is satisfied the requirements from the documentation:

    [https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_files.html](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_files.html)

  - Usage

    load_files functio have 2 dependencies: numpy and Bunch.

    We have run load\_files() function twice, one has using the _formats_ parameter, the other is not.

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a3/Images/19269-function-usage.png "Usage")

## Work Log

Here is the screenshot of the work log before and after the scrum.

Before Scrum:

![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a3/Images/scrum-start.png "Before Scrum")
