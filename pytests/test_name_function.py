from name_function import get_formatted_name

def test_first_last_name():
    '''Do names like 'Janis Joplin' work?'''
    formatted_name = project_two()
    assert formatted_name == 'Janis Joplin'
                                        
