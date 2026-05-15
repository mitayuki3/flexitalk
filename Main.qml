import QtQuick
import QtQuick.Window
import QtQuick.Controls
import QtQuick.Layouts

Window {
    width: 640
    height: 480
    visible: true
    title: qsTr("FlexiTalk")

    Rectangle {
        anchors.fill: parent
        color: palette.window

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
}
