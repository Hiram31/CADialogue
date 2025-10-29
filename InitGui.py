import os, sys, FreeCADGui


# Path to the folder where this script (InitGui.py) lives
mod_dir = os.path.dirname(os.path.abspath(__file__))

if mod_dir not in sys.path:
    sys.path.append(mod_dir)

# Import the command class from the assistant module
from CADialogue import CADAssistantCommand

class MyWorkbench(FreeCADGui.Workbench):
    MenuText = "CADialogue AI Assistant"
    ToolTip = "AI-Powered CAD Assistant Tools"
    Icon = ""

    def Initialize(self):
        FreeCADGui.addCommand('CAD_Assistant_Command', CADAssistantCommand())
        self.appendToolbar("AI-Powered CAD Assistant Tools", ["CAD_Assistant_Command"])
        self.appendMenu("AI-Powered CAD Assistant", ["CAD_Assistant_Command"])

    def Activated(self):
        pass

    def Deactivated(self):
        pass

FreeCADGui.addWorkbench(MyWorkbench())
