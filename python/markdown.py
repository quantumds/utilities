To delete all spaces in the file, replace ' +' with '' (quotes only for demonstration, please remove them). You need to have the checkbox "Regular expression" checked.

To remove all spaces and tabs, replace '[ \t]+' with '' (remove quotes).

This works for big files, too, where the solution of @Learner will be tiresome.

Regular expression box marked





To add a word, such as test, at the end of each line:

Type $ in the Find what textbox
Type test in the Replace with textbox
Place cursor in the first line of the file to ensure all lines are affected
Click Replace All button

Regular expression box marked



