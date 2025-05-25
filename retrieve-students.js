const admin = require("firebase-admin");
const fs = require("fs");
const path = require("path");

// Initialize Firebase with the service account
const serviceAccount = require("./research-app-9fff9-firebase-adminsdk-fbsvc-35cdf97b1e.json");

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});

// Get Firestore instance
const db = admin.firestore();

async function retrieveStudentsData() {
  try {
    // Get reference to the students collection
    const studentsRef = db.collection("students");

    // Retrieve all documents
    const snapshot = await studentsRef.get();

    if (snapshot.empty) {
      console.log("No students found in the collection.");
      return;
    }

    const studentsData = [];

    // Process each document
    snapshot.forEach((doc) => {
      studentsData.push({
        id: doc.id,
        ...doc.data(),
      });
    });

    // Convert to formatted JSON
    const jsonData = JSON.stringify(studentsData, null, 2);

    // Define output file path
    const outputPath = path.join(__dirname, "students-data.json");

    // Write data to file
    fs.writeFileSync(outputPath, jsonData);

    console.log(
      `Successfully retrieved ${studentsData.length} student records.`
    );
    console.log(`Data saved to: ${outputPath}`);
  } catch (error) {
    console.error("Error retrieving student data:", error);
  } finally {
    // Close the Firebase app when done
    admin.app().delete();
  }
}

// Execute the function
retrieveStudentsData();
