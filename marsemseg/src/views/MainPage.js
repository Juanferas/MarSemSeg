// import { Component } from 'react';

// function MainPage () {

//     const changeHandler = (event) => {
// 		// setSelectedFile(event.target.files[0]);
// 		// setIsSelected(true);
// 	};

// 	const handleSubmission = () => {
// 	};

//     return (
//             <div className="d-flex w-100">
//                 <h1> MarSemSeg </h1>
//                 <input type="file" name="file" onChange={changeHandler} />
//                 <button className=""> Upload file </button>
//                 <h3>Prediction:</h3>
//                 <img src="https://compote.slate.com/images/926e5009-c10a-48fe-b90e-fa0760f82fcd.png?width=1200&rect=680x453&offset=0x30" alt="stonks" />
//             </div>
//         );
// }

// export default MainPage;

import axios from 'axios';

import React,{Component} from 'react';
import CurvedText from '../components/CurvedText';

class MainPage extends Component {

	state = {
	    selectedFile: null,
        prediction: null
	};
	
	// File select
	onFileChange = event => {
	    this.setState({ selectedFile: event.target.files[0] });
	};
	
	// File upload
	onFileUpload = () => {
        const formData = new FormData();
        
        formData.append(
            "myFile",
            this.state.selectedFile,
            this.state.selectedFile.name
        );

        console.log(this.state.selectedFile);
        
        // Send file to backend
        axios.post("http://localhost:5000/file-upload", formData).then(response => {
            console.log(response.data);
          })
          .catch(function (error) {
            console.log(error);
          });
	};
	
	// Uploaded file info
	fileData = () => {
        if (this.state.selectedFile) {
            return (
                <div>
                    <br />
                    <h2>File Details:</h2>
                    
                    <p>File Name: {this.state.selectedFile.name}</p>
                    
                    <p>File Type: {this.state.selectedFile.type}</p>
                </div>
            );
        } else {
            return (
                <div>
                    <br />
                    <h4 style={{textShadow: '2px 2px 10px #FFFFFF'}}>Choose a file to which you want to apply Maritime Semantic Segmentation</h4>
                </div>
            );
        }
	};
	
	render() {
        return (
            <div>
                {/* <h1 className="font-face-bt" ><CurvedText style={{fontSize: '50px'}} text="MarSemSeg" /></h1> */}
                {/* <div className="border"> */}
                    <h1 className="font-face-bt" style={{fontSize: '200px', display: 'inline'}}> MarSemSeg</h1>
                    <img style={{marginLeft: '0.8rem', paddingBottom: '6rem'}}
                        src="https://cdn-icons-png.flaticon.com/512/4516/4516008.png"
                        alt="ancla"
                        width="130"
                        height="210"
                    />
                {/* </div> */}
                
                <div>
                    <label className="btn btn-info text-white shadow-lg" style={{textShadow: '2px 2px 8px #000000'}}>
                        <input type="file" name="file" accept="image/*, video/*" onChange={this.onFileChange} />
                        Select file
                    </label>
                </div>
                {this.fileData()}
                {this.state.selectedFile && <div>
                    <button className="btn btn-primary my-3" onClick={this.onFileUpload}> Upload </button>
                    {this.state.prediction && <h3> Prediction: </h3>}
                </div>}
            </div>
        );
	}
}

export default MainPage;
