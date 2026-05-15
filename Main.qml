import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: root
    width: 640
    height: 480
    visible: true
    title: qsTr("FlexiTalk")

    SystemPalette {
        id: systemPalette
        colorGroup: SystemPalette.Active
    }

    palette.window: systemPalette.window
    palette.windowText: systemPalette.windowText
    palette.base: systemPalette.base
    palette.text: systemPalette.text
    palette.button: systemPalette.button
    palette.buttonText: systemPalette.buttonText
    color: palette.window

    background: Rectangle {
        color: root.palette.window
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 12

        TextArea {
            id: ttsInputTextEdit
            Layout.fillWidth: true
            Layout.fillHeight: true
            placeholderText: "TTS入力文字列"
            wrapMode: TextEdit.Wrap
        }

        ComboBox {
            id: voiceFileSettingComboBox
            Layout.fillWidth: true
            model: ["voice_a.wav", "voice_b.wav", "voice_c.wav"]
        }

        Button {
            id: synthesizeButton
            text: "合成"
            Layout.fillWidth: true
        }
    }
}
