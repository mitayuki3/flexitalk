import QtQuick
import QtQuick.Window
import QtQuick.Controls

Window {
    width: 640
    height: 480
    visible: true
    title: qsTr("FlexiTalk (Hot Reload Enabled)")

    Rectangle {
        anchors.fill: parent
        color: "#f0f0f0"

        Column {
            anchors.centerIn: parent
            spacing: 20

            Text {
                text: "FlexiTalk QML Window"
                font.pixelSize: 24
                anchors.horizontalCenter: parent.horizontalCenter
            }

            Text {
                text: "このファイルを編集して保存すると、表示が更新されます。"
                font.pixelSize: 16
                color: "#666666"
                anchors.horizontalCenter: parent.horizontalCenter
            }

            Rectangle {
                width: 100
                height: 100
                color: "lightgreen"
                radius: 10
                anchors.horizontalCenter: parent.horizontalCenter

                Text {
                    anchors.centerIn: parent
                    text: "Edit Me!"
                }
            }
        }
    }
}
