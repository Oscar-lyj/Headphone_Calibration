import os
import numpy as np
from flask import render_template, request, redirect, url_for, flash, session, jsonify
import json
from werkzeug.utils import secure_filename
# from calibration import process_csv_module, process_image_module
from calibration.process_csv_module import process_csv
from calibration.process_image_module import interactive_image_selection

def init_routes(app):
    """Initialize routes for the Flask app."""
    
    UPLOAD_FOLDER = 'headphone_calibration\\app\\static\\uploads'
    ALLOWED_EXTENSIONS = {'csv', 'png'}

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/upload/csv', methods=['GET', 'POST'])    
    def upload_csv():
        """Handles CSV file uploads for calibration."""
        if request.method == 'POST':
            if 'csv_file' not in request.files:
                flash('No file part')
                return redirect(request.url)

            file = request.files['csv_file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                bands = int(request.form.get('bands', 10))  # Default to 10 if not provided
                reference_curve = request.form.get('reference_curve')
                # Check if reference_curve is numeric (0, 1, 2, 3) and convert to integer if so, otherwise handle 'custom'
                if reference_curve.isdigit():
                    reference_curve = int(reference_curve)
                else:
                    reference_curve = 'custom'

                reference_path = None

                # If the user chooses a custom reference curve
                if reference_curve == 'custom':
                    custom_curve_type = request.form.get('custom_curve_type')
                    if custom_curve_type == '5':
                        custom_file = request.files.get('custom_csv')
                        reference_curve = int(custom_curve_type)
                        if custom_file and allowed_file(custom_file.filename):
                            custom_filename = secure_filename(custom_file.filename)
                            custom_filepath = os.path.join(app.config['UPLOAD_FOLDER'], custom_filename)
                            custom_file.save(custom_filepath)
                            reference_path = custom_filepath  # Set the custom reference curve path
                        else:
                            flash("Invalid custom reference CSV file.")
                            return redirect(request.url)

                    elif custom_curve_type == '4':
                        reference_curve = int(custom_curve_type)
                        custom_file = request.files.get('custom_png')
                        ref_max_position = int(request.form.get("ref_max_position"))
                        ref_min_position = int(request.form.get("ref_min_position"))
                        reference_mode = int(request.form.get("ref_selection_mode"))
                        reference_sen = int(request.form.get("ref_sensitivity"))
                        if reference_mode == 1:
                            ref_selected_region_temp = request.form.get('ref_selected_region')
                            ref_selected_region_dict = json.loads(ref_selected_region_temp)
                            ref_selected_region = [ref_selected_region_dict['x1'], ref_selected_region_dict['y1'],
                                                        ref_selected_region_dict['x2'], ref_selected_region_dict['y2']]
                        elif reference_mode == 2:
                            ref_selected_region_temp = request.form.get('ref_rgb_selected')
                            ref_rgb_selected = json.loads(ref_selected_region_temp)
                            ref_selected_region = [ref_rgb_selected['r'], ref_rgb_selected['g'], ref_rgb_selected['b']]
                            
                        interactive_image_selection(custom_file, reference_mode, reference_sen, ref_max_position, ref_min_position, ref_selected_region, 2)
                        SAVE_PATH = "headphone_calibration\\app\\static\\uploads"     
                        reference_path = os.path.join(SAVE_PATH, "extracted_reference_curve.csv")                    

                # Process CSV file for EQ band calibration
                eq_bands, graph_path, object_curve, reference_curve, corrected_curve, freqs = process_csv(filepath, bands, reference_curve,
                                                                                                          reference_path=reference_path if reference_path is not None else None)
                graph_path_name = "uploads/corrected_response.png"
                return render_template('result_interactive.html', eq_bands=eq_bands, graph_path=graph_path_name, 
                                       object_curve=object_curve.tolist() if isinstance(object_curve, np.ndarray) else object_curve,
                                        reference_curve=reference_curve.tolist() if isinstance(reference_curve, np.ndarray) else reference_curve,
                                        corrected_curve=corrected_curve.tolist() if isinstance(corrected_curve, np.ndarray) else corrected_curve,
                                        freq_labels = freqs.tolist() if isinstance(corrected_curve, np.ndarray) else freqs)

            else:
                flash('Invalid file type. Please upload a valid CSV file.')
                return redirect(request.url)

        return render_template('upload_csv.html')
    
    @app.route('/process_block_selection', methods=['POST'])
    def process_block_selection():
        """Handles the bounding box selection from the block selection page."""
        data = request.get_json()

        if 'selectedRegion' in data:
            selected_region = data['selectedRegion']

            # Store the coordinates in the session for later use
            session['selected_region'] = selected_region

            return jsonify({
                "status": "success", 
                "message": "Region selected",
                "selected_region": selected_region
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Invalid selection data"
            }), 400
            
    @app.route('/process_color_selection', methods=['POST'])
    def process_color_selection():
        """Handles the color selection from the color selection page."""
        data = request.get_json()

        if 'color' in data and 'sensitivity' in data:
            selected_color = data['color']
            # sensitivity = data['sensitivity']

            # Store the selected color in the session as selected_region
            session['selected_region'] = selected_color
            # session['sensitivity'] = sensitivity

            return jsonify({"status": "success", "message": "Color selection submitted", "selected_region": selected_color}), 200
        else:
            return jsonify({"status": "error", "message": "Invalid selection data"}), 400
        
    @app.route('/upload/png', methods=['GET', 'POST'])
    def upload_png():
        """Handles PNG image uploads for frequency response extraction."""
        if request.method == 'POST':
            if 'png_file' not in request.files:
                flash('No file part')
                return redirect(request.url)

            file = request.files['png_file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                mode = request.form.get('selection_mode')
                sensitivity = int(request.form.get('sensitivity', 60))  # Default to 60
                
                bands = int(request.form.get('bands', 10))  # Default to 10 if not provided
                reference_curve= request.form.get('reference_curve')
                min_pos = int(request.form.get("min_position"))
                max_pos = int(request.form.get("max_position"))
                
                # storing the variables
                
                session['filepath'] = filepath
                session['mode'] = mode
                session['sensitivity'] = sensitivity
                session['max_pos'] = max_pos
                session['min_pos'] = min_pos
                session['bands'] = bands
                session['reference_curve'] = reference_curve
                # If the user chooses a custom reference curve
                if reference_curve == 'custom':
                    custom_curve_type = request.form.get('custom_curve_type')
                    session["custom_curve_type"] = custom_curve_type
                    if custom_curve_type == "5":
                        custom_file = request.files.get('custom_csv')
                        if custom_file and allowed_file(custom_file.filename):
                            custom_filename = secure_filename(custom_file.filename)
                            custom_filepath = os.path.join(app.config['UPLOAD_FOLDER'], custom_filename)
                            custom_file.save(custom_filepath)
                            session["custom_file"] = custom_filepath  # Save the path instead of the file
                    elif custom_curve_type == "4":
                        custom_file = request.files.get('custom_png')
                        session["ref_max_position"] = int(request.form.get("ref_max_position"))
                        session["ref_min_position"] = int(request.form.get("ref_min_position"))
                        reference_mode = int(request.form.get("ref_selection_mode"))
                        session["reference_mode"] = reference_mode
                        session["reference_sen"] = int(request.form.get("ref_sensitivity"))
                        
                        if reference_mode == 1:
                            ref_selected_region_temp = request.form.get('ref_selected_region')
                            ref_selected_region_dict = json.loads(ref_selected_region_temp)
                            ref_selected_region = [ref_selected_region_dict['x1'], ref_selected_region_dict['y1'],
                                                        ref_selected_region_dict['x2'], ref_selected_region_dict['y2']]
                            session["ref_selected_region"] = ref_selected_region
                        elif reference_mode == 2:
                            ref_selected_region_temp = request.form.get('ref_rgb_selected')
                            ref_rgb_selected = json.loads(ref_selected_region_temp)
                            ref_selected_region = [ref_rgb_selected['r'], ref_rgb_selected['g'], ref_rgb_selected['b']]
                            session["ref_selected_region"] = ref_selected_region
                            
                        if custom_file and allowed_file(custom_file.filename):
                            custom_filename = secure_filename(custom_file.filename)
                            custom_filepath = os.path.join(app.config['UPLOAD_FOLDER'], custom_filename)
                            custom_file.save(custom_filepath)
                            session["custom_file"] = custom_filepath  # Save the path instead of the file

                if mode == "1":  # Block Selection Mode
                    return render_template('block_selection.html', image_path=filename)

                elif mode == "2":  # Color Selection Mode
                    return render_template('color_selection.html', image_path=filename, sensitivity=sensitivity)
            
                else:
                    flash('Invalid file type. Please upload a valid PNG file.')
                    return redirect(request.url)

        return render_template('upload_png.html')
    

    @app.route('/calibrate', methods=['GET', 'POST'])
    def calibrate():
        """Handles the calibration process after image selection."""
        
        # If it's a GET request, ensure that session data is available
        reference_path = None
        if request.method == 'GET':           
            # If the session data exists, proceed with the calibration           
            filepath = session.get('filepath')
            mode = int(session.get('mode', ))  # Default to mode 1 if not set
            sensitivity = int(session.get('sensitivity', 60))
            max_pos = int(session.get('max_pos', 100))
            min_pos = int(session.get('min_pos', 80))
            bands = int(session.get('bands', 10))
            reference_curve = session.get('reference_curve')
            if reference_curve.isdigit():
                reference_curve = int(reference_curve)
            else:
                reference_curve = 'custom'
                custom_curve_type = session.get("custom_curve_type")

                if custom_curve_type == '5':
                    custom_filepath = session.get("custom_file")
                    reference_curve = int(custom_curve_type)
                    reference_path = custom_filepath  # Set the custom reference curve path

                elif custom_curve_type == '4':
                    reference_curve = int(custom_curve_type)
                    custom_filepath = session.get("custom_file")
                    ref_max_position = session.get("ref_max_position")
                    ref_min_position = session.get("ref_min_position")
                    reference_mode = session.get("reference_mode")
                    reference_sen = session.get("reference_sen")
                    if reference_mode == 1:
                        ref_selected_region = session.get("ref_selected_region")
                    elif reference_mode == 2:
                        ref_selected_region = session.get("ref_selected_region")
                                                   
                    interactive_image_selection(custom_filepath, reference_mode, reference_sen, ref_max_position, ref_min_position, ref_selected_region, 2)
                    SAVE_PATH = "headphone_calibration\\app\\static\\uploads"     
                    reference_path = os.path.join(SAVE_PATH, "extracted_reference_curve.csv")   
    
            selected_region_temp = session.get('selected_region')
            if mode == 1:
                selected_region = selected_region_temp
            elif mode == 2:
                selected_region = [selected_region_temp['b'], selected_region_temp['g'], selected_region_temp['r']]
                
            if not selected_region:
                flash("No region selected. Please go back and select a region.")
                return redirect(url_for('upload_png'))

            # Process the image and perform calibration
            interactive_image_selection(filepath, mode, sensitivity, max_pos, min_pos, selected_region,1)
            extracted_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], "extracted_curve.csv")

            eq_bands, graph_path, object_curve, reference_curve, corrected_curve, freqs = process_csv(extracted_csv_path, bands, reference_curve,
                                                                                                      reference_path=reference_path if reference_path is not None else None)
            graph_path_name = "uploads/corrected_response.png"

            # Render result page with EQ bands and corrected graph
            return render_template('result_interactive.html', eq_bands=eq_bands, graph_path=graph_path_name, 
                                    object_curve=object_curve.tolist() if isinstance(object_curve, np.ndarray) else object_curve,
                                    reference_curve=reference_curve.tolist() if isinstance(reference_curve, np.ndarray) else reference_curve,
                                    corrected_curve=corrected_curve.tolist() if isinstance(corrected_curve, np.ndarray) else corrected_curve,
                                    freq_labels = freqs.tolist() if isinstance(corrected_curve, np.ndarray) else freqs)
    
        # If it's a POST request, the same calibration logic can be applied
        '''
        if request.method == 'POST':
            selected_region = session.get('selected_region')
            mode = int(session.get('mode'))
            sensitivity = int(session.get('sensitivity'))
            max_pos = int(session.get('max_pos'))
            min_pos = int(session.get('min_pos'))
            filepath = session.get('filepath')
            bands = int(session.get('bands'))
            reference_curve = session.get('reference_curve')

            if not selected_region:
                flash("Region selection is missing.")
                return redirect(url_for('upload_png'))

            # Extract frequency response using the selected region
            extracted_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], "extracted_curve.csv")
            interactive_image_selection(filepath, mode, sensitivity, max_pos, min_pos, selected_region)

            # Process CSV to get EQ bands and generate graph
            eq_bands, graph_path = process_csv(extracted_csv_path, bands, reference_curve)
            graph_path_name = "uploads/corrected_response.png"

            # Render result page with EQ bands and corrected graph
            return render_template('result.html', eq_bands=eq_bands, graph_path=graph_path_name)
        '''
